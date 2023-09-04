# Adapted from
# https://github.com/recursionpharma/gflownet/blob/trunk/src/gflownet/models/graph_transformer.py

import torch
from torch import nn
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
import torch_geometric.data as gd

from src.graph_building_env import GraphStateType, NodeStateType
from src.distributions import (
    GraphActionCategorical,
    AuxInitActionCategorical,
    InitActionCategorical,
    NodeLevelActionCategorical,
    EdgeLevelActionCategorical,
)


def transpose(permute_tensor):
    p = torch.empty_like(permute_tensor)
    p[permute_tensor] = torch.arange(len(permute_tensor))
    return p


def mlp(n_in, n_hid, n_out, n_layer=1, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(
        *sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1]
    )


class GraphModel(nn.Module):
    def __init__(
        self, num_node_types, num_edge_types, emb_dim=64, num_layers=3, num_heads=2
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(
                            emb_dim, emb_dim, num_layers=1, aggr="add", norm=None
                        ),
                        gnn.TransformerConv(
                            emb_dim * 2, emb_dim, edge_dim=emb_dim, heads=num_heads
                        ),
                        nn.Linear(num_heads * emb_dim, emb_dim),
                        gnn.LayerNorm(emb_dim, affine=False),
                        mlp(emb_dim, emb_dim * 4, emb_dim, 1),
                        gnn.LayerNorm(emb_dim, affine=False),
                    ]
                    for i in range(num_layers)
                ],
                [],
            )
        )
        self.positional_emb = gnn.PositionalEncoding(emb_dim)
        self.node_type_emb = nn.Embedding(num_node_types + 1, emb_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types, emb_dim)
        self.node_state_type_emb = nn.Embedding(
            4, emb_dim
        )  # frontier, edge_source, node_source, ancestor
        self.virtual_node_emb = nn.Embedding(1, emb_dim)

        self.init_mlp = mlp(emb_dim, emb_dim, num_node_types)
        self.node_level_mlp = mlp(emb_dim, emb_dim, num_edge_types * num_node_types + 1)
        self.edge_level_mlp = mlp(emb_dim, emb_dim, num_edge_types)
        self.edge_stop_mlp = mlp(emb_dim, emb_dim, 1)

        self.aux_init_mlp = mlp(emb_dim, emb_dim, 1)
        self.aux_node_level_mlp = mlp(
            emb_dim, emb_dim, num_edge_types * num_node_types + 1
        )
        self.aux_edge_level_mlp = mlp(emb_dim, emb_dim, num_edge_types)

    def forward(self, states):
        g = self.collate(states)
        assert (
            g.graph_state_type == GraphStateType.Terminal.value
        ).sum().item() == 0, "Terminal state may cause errors"

        embeddings = self.embed(g)

        glob = embeddings["graph_embeddings"]
        ne_emb = embeddings["non_edge_embeddings"]

        init_mask = g.graph_state_type == GraphStateType.Initial.value
        node_mask = g.graph_state_type == GraphStateType.NodeLevel.value
        edge_mask = g.graph_state_type == GraphStateType.EdgeLevel.value

        init_batch = init_mask.nonzero().flatten()
        node_batch = node_mask.nonzero().flatten()
        edge_batch = edge_mask.nonzero().flatten()

        init_logits = self.init_mlp(glob[init_mask])
        node_logits = self.node_level_mlp(glob[node_mask])
        edge_logits = self.edge_level_mlp(ne_emb)
        stop_logits = self.edge_stop_mlp(glob[edge_mask])

        loc = torch.cat([init_batch, node_batch, edge_batch])
        order = transpose(loc)

        init_cat = InitActionCategorical(init_logits)
        node_cat = NodeLevelActionCategorical(node_logits, self.num_edge_types)
        edge_cat = EdgeLevelActionCategorical(
            stop_logits,
            edge_logits,
            edge_batch,
            g.non_edge_batch,
            self.num_edge_types,
        )

        return GraphActionCategorical(init_cat, node_cat, edge_cat, order)

    def aux_forward(self, states, cond_states):
        g = self.collate(states, cond_states)
        embeddings = self.embed(g)

        glob = embeddings["graph_embeddings"]
        ne_emb = embeddings["non_edge_embeddings"]
        node_emb = embeddings["node_embeddings"]

        init_mask = g.graph_state_type == GraphStateType.Initial.value
        node_mask = g.graph_state_type == GraphStateType.NodeLevel.value
        edge_mask = g.graph_state_type == GraphStateType.EdgeLevel.value

        init_batch = init_mask.nonzero().flatten()
        node_batch = node_mask.nonzero().flatten()
        edge_batch = edge_mask.nonzero().flatten()

        init_graph_idx = set(init_batch.tolist())
        init_node_mask = torch.tensor(
            [idx.item() in init_graph_idx for idx in g.batch], dtype=torch.bool
        )
        init_logits = self.aux_init_mlp(node_emb[init_node_mask])
        node_logits = self.aux_node_level_mlp(glob[node_mask])
        edge_logits = self.aux_edge_level_mlp(ne_emb)
        edge_logits[~g.non_edge_index_mask] = -float("inf")
        # print(edge_logits.shape)
        # print(edge_logits[~g.non_edge_index_mask].shape)

        loc = torch.cat([init_batch, node_batch, edge_batch])
        order = transpose(loc)

        # force stop prob=100%
        node_logits[g.node_type_mask.sum(dim=1) == 0.0, 0] = +10000
        # return g.node_type_mask, node_logits
        init_cat = AuxInitActionCategorical(init_logits, g.batch[init_node_mask])
        node_cat = NodeLevelActionCategorical(
            -10000 * (~g.node_type_mask) + node_logits, self.num_edge_types
        )
        edge_cat = EdgeLevelActionCategorical(
            -10000 * torch.ones_like(edge_batch, dtype=torch.float),
            edge_logits,
            edge_batch,
            g.non_edge_batch,
            self.num_edge_types,
        )

        return GraphActionCategorical(init_cat, node_cat, edge_cat, order)

    def embed(self, g):
        n_emb = self.node_type_emb(g.node_type)
        s_emb = self.node_state_type_emb(
            g.node_state_type.clip(0, NodeStateType.Frontier)
        )
        p_emb = self.positional_emb(g.node_state_type)
        x = n_emb + s_emb + p_emb
        e = self.edge_type_emb(g.edge_type)

        # can be used to condition graph-level features.
        cond = torch.zeros(g.num_graphs, dtype=torch.long, device=self.device)
        c = self.virtual_node_emb(cond)

        # Append the virtual node embedding
        x_aug = torch.cat([x, c], 0)

        # Virtual node is connected to every node.
        u, v = torch.arange(g.num_nodes, device=self.device), g.batch + g.num_nodes
        aug_edge_index = torch.cat(
            [g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1
        )
        e_p = torch.zeros((g.num_nodes * 2, e.shape[1]), device=self.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat(
            [g.batch, torch.arange(g.num_graphs, device=self.device)], 0
        )

        node_emb, glob = self.gnn_forward(g, x_aug, aug_edge_index, aug_e, aug_batch)
        ne_row, ne_col = g.non_edge_index
        ne_emb = node_emb[ne_row] + node_emb[ne_col]
        return dict(
            node_embeddings=node_emb,
            graph_embeddings=glob,
            non_edge_embeddings=ne_emb,
        )

    def gnn_forward(self, g, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2 = self.graph2emb[i * 6 : (i + 1) * 6]
            x_norm = norm1(x, batch)
            agg = gen(x_norm, edge_index, edge_attr)
            l_h = linear(trans(torch.cat([x_norm, agg], 1), edge_index, edge_attr))
            x = x + ff(norm2(l_h, batch))

        o_final = x[: -g.num_graphs]  # final node embeddings
        v_final = x[-g.num_graphs :]  # virtual node embeddings
        if len(o_final) == 0:  # edge case: empty graph
            glob = v_final
        else:
            glob = gnn.global_mean_pool(o_final, g.batch) + v_final
        return o_final, glob

    def collate(self, states, cond_states=None):
        data_list = [s.to_tensor_graph() for s in states]
        non_edge_index_mask, node_type_masks = None, None
        if cond_states is not None:
            node_dim = self.num_edge_types * self.num_node_types + 1
            edge_masks = [torch.tensor([], dtype=torch.bool)]
            node_masks = [torch.tensor([], dtype=torch.bool).reshape(0, node_dim)]
            for i, (s, c) in enumerate(zip(states, cond_states)):
                d = data_list[i]
                if s.state_type == GraphStateType.Initial:
                    data_list[i] = cond_states[i].to_tensor_graph()
                elif s.state_type == GraphStateType.NodeLevel:
                    # node mask
                    nm = torch.zeros(node_dim, dtype=torch.bool)
                    cond_set = [
                        1 + et + nt * self.num_edge_types
                        for nt, et in s.cond_node_targets(c)
                    ]
                    nm[cond_set] = True
                    node_masks.append(nm.unsqueeze(0))
                elif s.state_type == GraphStateType.EdgeLevel:
                    # edge mask
                    tgt, cond_set = d.non_edge_index[1], s.cond_edge_targets(c)
                    em = torch.tensor(
                        [t in cond_set for t in tgt.tolist()], dtype=torch.bool
                    )
                    edge_masks.append(em)

            non_edge_index_mask = torch.cat(edge_masks, dim=0)
            node_type_masks = torch.cat(node_masks, dim=0)

        bg = gd.Batch.from_data_list(data_list)
        non_edge_batch = [g.non_edge_index.shape[1] for g in data_list]
        bg.non_edge_batch = torch.arange(bg.num_graphs).repeat_interleave(
            torch.LongTensor(non_edge_batch)
        )
        bg.graph_state_type = torch.tensor([s.state_type.value for s in states])
        bg.non_edge_index_mask = non_edge_index_mask
        bg.node_type_mask = node_type_masks

        return bg

    @property
    def device(self):
        return next(self.parameters()).device
