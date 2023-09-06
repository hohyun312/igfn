# Adapted from
# https://github.com/recursionpharma/gflownet/blob/trunk/src/gflownet/models/graph_transformer.py

import torch
from torch import nn
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
import torch_geometric.data as gd

from src.graph_building_env import GraphStateType, NodeStateType


from src.distributions import ActionCategorical


def add_state_type_location(g):
    g.is_init = g.state_type == GraphStateType.Initial.value
    g.is_nodelv = g.state_type == GraphStateType.NodeLevel.value
    g.is_edgelv = g.state_type == GraphStateType.EdgeLevel.value
    g.is_terminal = g.state_type == GraphStateType.Terminal.value

    g.init_loc = g.is_init.nonzero().flatten()
    g.nodelv_loc = g.is_nodelv.nonzero().flatten()
    g.edgelv_loc = g.is_edgelv.nonzero().flatten()
    g.terminal_loc = g.is_terminal.nonzero().flatten()


def collate(states, cond_states=None):
    if cond_states is None:
        data_list = [s.to_tensor_graph() for s in states]
    else:
        data_list = [None] * len(states)
        for i in range(len(states)):
            s, c = states[i], cond_states[i]
            if s.state_type.value == GraphStateType.Initial.value:
                data_list[i] = c.to_tensor_graph()
                data_list[i].wl_node_index = torch.LongTensor(c.unique_nodes())
            else:
                data_list[i] = s.to_tensor_graph(c)
                data_list[i].wl_node_index = torch.LongTensor([])

    graphs = gd.Batch.from_data_list(data_list)

    # extra information
    graphs.num_non_edge = torch.LongTensor(
        [g.non_edge_index.shape[1] for g in data_list]
    )
    graphs.state_type = torch.tensor([s.state_type.value for s in states])
    add_state_type_location(graphs)
    return graphs


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
        self.nodelv_mlp = mlp(emb_dim, emb_dim, num_edge_types * num_node_types + 1)
        self.edgelv_tgt_mlp = mlp(emb_dim, emb_dim, num_edge_types)
        self.edgelv_stop_mlp = mlp(emb_dim, emb_dim, 1)

        self.aux_init_mlp = mlp(emb_dim, emb_dim, 1)
        self.aux_nodelv_mlp = mlp(emb_dim, emb_dim, num_edge_types * num_node_types + 1)
        self.aux_edgelv_tgt_mlp = mlp(emb_dim, emb_dim, num_edge_types)

    def forward(self, states):
        g = collate(states)

        assert len(g.terminal_loc) == 0, "Terminal state may cause errors"

        embeddings = self.embed(g)

        glob = embeddings["graph_embeddings"]
        ne_emb = embeddings["non_edge_embeddings"]

        init_logits = self.init_mlp(glob[g.is_init])
        nodelv_logits = self.nodelv_mlp(glob[g.is_nodelv])
        edge_target_logits = self.edgelv_tgt_mlp(ne_emb)
        edge_stop_logits = self.edgelv_stop_mlp(glob[g.is_edgelv])

        edgelv_logits = torch.cat(
            [edge_stop_logits.flatten(), edge_target_logits.flatten()], 0
        )
        edge_target_indices = torch.arange(g.num_graphs).repeat_interleave(
            g.num_non_edge * self.num_edge_types
        )
        edgelv_indices = torch.cat([g.edgelv_loc, edge_target_indices])

        locations = torch.cat([g.init_loc, g.nodelv_loc, g.edgelv_loc])

        return ActionCategorical(
            init_logits,
            nodelv_logits,
            edgelv_logits,
            edgelv_indices,
            locations,
            self.num_edge_types,
        )

    def aux_forward(self, states, cond_states):
        if not isinstance(cond_states, list):
            cond_states = [cond_states] * len(states)

        g = collate(states, cond_states)
        embeddings = self.embed(g)

        glob = embeddings["graph_embeddings"]
        ne_emb = embeddings["non_edge_embeddings"]
        node_emb = embeddings["node_embeddings"]

        init_node_mask = torch.any(g.batch[:, None] == g.init_loc, dim=1)
        init_logits = self.aux_init_mlp(node_emb[init_node_mask]).flatten()
        nodelv_logits = self.aux_nodelv_mlp(glob[g.is_nodelv])
        edge_target_logits = self.aux_edgelv_tgt_mlp(ne_emb)

        # mask init
        init_mask = torch.zeros(max(g.wl_node_index, default=-1) + 1, dtype=torch.bool)
        init_mask[g.wl_node_index] = True
        init_logits = torch.where(init_mask, 0, -torch.inf) + init_logits

        # mask nodelv
        nodelv_mask = self.make_node_inv_mask(g, states, cond_states)
        nodelv_mask = torch.where(nodelv_mask, -torch.inf, 0)
        nodelv_logits = nodelv_mask + nodelv_logits

        # mask edgelv
        edgelv_num_non_edge = g.num_non_edge[
            g.state_type == GraphStateType.EdgeLevel.value
        ]
        edge_stop_logits = torch.where(edgelv_num_non_edge == 0, 0, -torch.inf)
        edgelv_logits = torch.cat([edge_stop_logits, edge_target_logits.flatten()], 0)
        edge_target_indices = torch.arange(g.num_graphs).repeat_interleave(
            g.num_non_edge * self.num_edge_types
        )
        edgelv_indices = torch.cat([g.edgelv_loc, edge_target_indices])

        locations = torch.cat([g.init_loc, g.nodelv_loc, g.edgelv_loc])

        return ActionCategorical(
            init_logits=init_logits,
            nodelv_logits=nodelv_logits,
            edgelv_logits=edgelv_logits,
            edgelv_indices=edgelv_indices,
            logits_locations=locations,
            num_edge_types=self.num_edge_types,
            init_indices=g.batch[init_node_mask],
        )

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

    def make_node_inv_mask(self, g, states, cond_states):
        """
        node-level mask (invalid actions are marked `True`)
        """
        N = (g.state_type == GraphStateType.NodeLevel.value).sum().item()
        D = self.num_node_types * self.num_edge_types + 1
        node_mask = torch.ones(N, D, dtype=torch.bool)

        for i, k in enumerate(g.nodelv_loc):
            k = k.item()
            s, c = states[k], cond_states[k]

            inward_adj = s.graph[s.node_source]
            outward_adj = c.graph[s.node_source]
            targets = set(outward_adj) - set(inward_adj)
            n, e = c.graph.nodes, outward_adj
            idx = [0] + [
                n[t]["node_type"] * self.num_edge_types + e[t]["edge_type"] + 1
                for t in targets
            ]
            node_mask[i, idx] = False

        return node_mask

    @property
    def device(self):
        return next(self.parameters()).device
