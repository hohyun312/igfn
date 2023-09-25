from typing import List

import torch
from torch import nn
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
import torch_geometric.data as gd

import src.index_utils as index_utils
from src.distributions import ActionCategorical, DefaultCategorical, FlatCategorical
from src.graph_env import GraphEnv
from src.containers import GraphState, SortedStates


def mlp(n_in, n_hid, n_out, n_layer=1, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(
        *sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1]
    )


class PositionalEncoding(gnn.PositionalEncoding):
    def forward(self, x):
        output = super().forward(x)
        output[x == -1] = 0.0
        return output


class TwoHeadMLP(nn.Module):
    def __init__(self, head1_param, head2_param):
        super().__init__()
        self.head1 = mlp(*head1_param)
        self.head2 = mlp(*head2_param)

    def forward(self, x1, x2):
        logit1 = self.head1(x1).flatten()
        logit2 = self.head2(x2).flatten()
        return torch.cat([logit1, logit2], 0)


class GraphTransformer(nn.Module):
    def __init__(self, emb_dim: int = 64, num_layers: int = 3, num_heads: int = 2):
        super().__init__()
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
                    for _ in range(num_layers)
                ],
                [],
            )
        )

    def forward(self, g):
        x, e, c = g.x, g.edge_attr, g.cond

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
        h_aug = self.gnn_forward(x_aug, aug_edge_index, aug_e, aug_batch)
        n_emb = h_aug[: -g.num_graphs]  # final node embeddings
        v_emb = h_aug[-g.num_graphs :]  # virtual node embeddings

        if len(n_emb) == 0:  # edge case: empty graph
            glob = v_emb
        else:
            glob = gnn.global_mean_pool(n_emb, g.batch) + v_emb

        ne_row, ne_col = g.non_edge_index
        ne_emb = n_emb[ne_row] + n_emb[ne_col]

        return dict(
            node_embeddings=n_emb,
            graph_embeddings=glob,
            non_edge_embeddings=ne_emb,
        )

    def gnn_forward(self, x, edge_index, edge_attr, batch):
        # Run the graph transformer forward
        for i in range(self.num_layers):
            gen, trans, linear, norm1, ff, norm2 = self.graph2emb[i * 6 : (i + 1) * 6]
            x_norm = norm1(x, batch)
            agg = gen(x_norm, edge_index, edge_attr)
            l_h = linear(trans(torch.cat([x_norm, agg], 1), edge_index, edge_attr))
            x = x + ff(norm2(l_h, batch))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class GraphPolicy(nn.Module):
    def __init__(
        self, env: GraphEnv, emb_dim: int = 64, num_layers: int = 3, num_heads: int = 2
    ):
        super().__init__()
        self.env = env
        self.transf = GraphTransformer(emb_dim, num_layers, num_heads)
        self.logZ = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.emb = nn.ModuleDict(
            {
                "frontier_order": PositionalEncoding(emb_dim),
                "node_type": nn.Embedding(env.num_node_types + 1, emb_dim),
                "edge_type": nn.Embedding(env.num_edge_types, emb_dim),
                "node_state_type": nn.Embedding(4, emb_dim),
                "virtual_node": nn.Embedding(1, emb_dim),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "init": mlp(emb_dim, emb_dim, env.num_node_types),
                "nodelv": mlp(
                    emb_dim, emb_dim, env.num_edge_types * env.num_node_types + 1
                ),
                "edgelv": TwoHeadMLP(
                    [emb_dim, emb_dim, 1],
                    [emb_dim, emb_dim, env.num_edge_types],
                ),
            }
        )
        self.cond_mlp = nn.ModuleDict(
            {
                "init": mlp(emb_dim, emb_dim, 1),
                "nodelv": mlp(emb_dim, emb_dim, 1),
                "edgelv": mlp(emb_dim, emb_dim, env.num_edge_types),
            }
        )

    def get_embeddings(self, g):
        g.x = (
            self.emb["node_type"](g.node_type)
            + self.emb["node_state_type"](g.node_state_type)
            + self.emb["frontier_order"](g.frontier_order)
        )
        g.edge_attr = self.emb["edge_type"](g.edge_type)
        g.cond = self.emb["virtual_node"](
            torch.zeros(g.num_graphs, dtype=torch.long, device=self.device)
        )
        embeddings = self.transf(g)
        return embeddings

    def forward(self, states: List[GraphState]):
        sorted_states = SortedStates(states)
        g = self.env.collate(sorted_states)

        assert sorted_states.sizes[-1] == 0, "Terminal state may cause errors"
        embeddings = self.get_embeddings(g)

        _, i, j, k, _ = g.sptr
        init_glob = embeddings["graph_embeddings"][:i]
        nodelv_glob = embeddings["graph_embeddings"][i:j]
        edgelv_glob = embeddings["graph_embeddings"][j:k]
        ne_emb = embeddings["non_edge_embeddings"]

        init_logits = self.mlp["init"](init_glob)
        nodelv_logits = self.mlp["nodelv"](nodelv_glob)
        edgelv_logits = self.mlp["edgelv"](edgelv_glob, ne_emb)

        edgelv_stop_index = torch.arange(k - j)
        edgelv_tgt_index = edgelv_stop_index.repeat_interleave(
            g.num_non_edges[j:k] * self.env.num_edge_types
        )
        edgelv_index = torch.cat([edgelv_stop_index, edgelv_tgt_index])

        init_cat = DefaultCategorical(logits=init_logits)
        nodelv_cat = DefaultCategorical(logits=nodelv_logits)
        edgelv_cat = FlatCategorical(logits=edgelv_logits, indices=edgelv_index)

        return ActionCategorical(
            init_cat,
            nodelv_cat,
            edgelv_cat,
            self.env.num_edge_types,
            sorted_states.indices(),
        )

    def cond_forward(self, states):
        sorted_states = SortedStates(states)
        g = self.env.collate(sorted_states)

        assert sorted_states.sizes[-1] == 0, "Terminal state may cause errors"
        embeddings = self.get_embeddings(g)

        _, i, j, k, _ = g.nptr
        init_nemb = embeddings["node_embeddings"][:i]
        nodelv_nemb = embeddings["node_embeddings"][i:j]
        edgelv_nemb = embeddings["node_embeddings"][j:k]

        init_targets, init_node_types, init_mappings = get_init_targets(
            sorted_states.initial
        )
        nodelv_targets, *_ = get_nodelv_targets(sorted_states.node_level)
        i, j = get_edgelv_targets(sorted_states.edge_level)

        init_logits = self.cond_mlp["init"](init_nemb[init_targets])
        nodelv_logits = self.cond_mlp["nodelv"](nodelv_nemb[nodelv_targets])
        edgelv_logits = self.cond_mlp["edgelv"](edgelv_nemb[i] + edgelv_nemb[j])

    def cond_forward(self, states: List[GraphState], cond_states: List[GraphState]):
        if not isinstance(cond_states, list):
            cond_states = [cond_states] * len(states)

        g = self.env.collate(states, cond_states)
        embeddings = self.embed(g)

        glob = embeddings["graph_embeddings"]
        ne_emb = embeddings["non_edge_embeddings"]
        n_emb = embeddings["node_embeddings"]

        # init mask
        is_init_node = torch.isin(g.batch, g.init_index)
        wl_mask = index_utils.to_mask(g.wl_node_index, g.num_nodes)
        is_valid_init = is_init_node & wl_mask

        # node mask
        is_valid_node = ...

        # forward
        init_logits = self.cond_mlp["init"](n_emb[is_valid_init]).flatten()
        nodelv_logits = self.cond_mlp["nodelv"](n_emb[is_valid_node]).flatten()
        edge_target_logits = self.cond_mlp["edgelv"](ne_emb)

        # mask nodelv
        nodelv_mask = self.make_node_mask(g, states, cond_states)
        nodelv_logits = nodelv_mask + nodelv_logits

        # mask edgelv
        # mask edges not in conditional graph
        # mask stop actions if conditional graph is not stop action
        edge_mask, stop_mask = self.make_edge_mask(g, states, cond_states)
        edge_target_logits = edge_mask + edge_target_logits
        # concat stop actions
        edgelv_logits = torch.cat([stop_mask, edge_target_logits.flatten()], 0)
        edge_target_indices = torch.arange(g.num_graphs).repeat_interleave(
            g.num_non_edges * self.env.num_edge_types
        )
        edgelv_indices = torch.cat([g.edgelv_index, edge_target_indices])

        locations = torch.cat([g.init_index, g.nodelv_index, g.edgelv_index])

        cond_info = dict(
            init_indices=g.batch[is_valid_init],
            node_types=g.node_type[is_valid_init],
            init_node_labels=g.node_label[is_valid_init],
            # nodelv_node_labels=g.node_label[is_valid_init]
        )

        return ActionCategorical(
            init_logits=init_logits,
            nodelv_logits=nodelv_logits,
            edgelv_logits=edgelv_logits,
            edgelv_indices=edgelv_indices,
            logits_locations=locations,
            num_edge_types=self.env.num_edge_types,
            cond_info=cond_info,
        )

    def make_node_mask(self, g, states, conds):
        """
        node-level mask (invalid actions get -torch.inf)
        """
        nodelv_mask = torch.stack(
            [
                self.env.conditional_node_mask(states[k.item()], conds[k.item()])
                for k in g.nodelv_index
            ],
            dim=0,
        )
        return torch.where(nodelv_mask, 0, -torch.inf)

    def make_node_mask(self, g, states, cond_states):
        """
        node-level mask (valid actions are marked `True`)
        """
        N = len(g.nodelv_index)
        D = self.env.num_node_types * self.env.num_edge_types + 1
        node_mask = torch.zeros(N, D, dtype=torch.bool)

        for i, k in enumerate(g.nodelv_index):
            state, cond = states[k.item()], cond_states[k.item()]

            inward_adj = state.graph[state.node_source]
            outward_adj = cond.graph[state.node_source]
            targets = set(outward_adj) - set(inward_adj)
            n, e = cond.graph.nodes, outward_adj
            stop = [] if targets else [0]
            idx = stop + [
                n[t]["node_type"] * self.env.num_edge_types + e[t]["edge_type"] + 1
                for t in targets
            ]
            node_mask[i, idx] = True

        return torch.where(node_mask, 0, -torch.inf)

    def make_edge_mask(self, g, states, cond_states):
        """
        edge-level mask (invalid actions get -torch.inf)
        """
        edge_mask = [
            torch.tensor([], dtype=torch.bool).reshape(0, self.env.num_edge_types)
        ]
        stop_mask = torch.ones(len(g.edgelv_index), dtype=torch.bool)
        for i, k in enumerate(g.edgelv_index):
            state, cond = states[k.item()], cond_states[k.item()]
            tgt = state.edge_targets
            cond_edges = set(cond.graph[state.edge_source])
            valid_targets = [(i, t) for i, t in enumerate(tgt) if t in cond_edges]

            mask = torch.zeros(len(tgt), self.env.num_edge_types, dtype=torch.bool)
            if valid_targets:
                valid_tgt_index, valid_tgt = zip(*valid_targets)
                valid_tgt_index = torch.LongTensor(valid_tgt_index)
                valid_type_index = torch.LongTensor(
                    [
                        cond.graph.edges[state.edge_source, i]["edge_type"]
                        for i in valid_tgt
                    ]
                )
                mask[valid_tgt_index, valid_type_index] = True
                stop_mask[i] = False

            edge_mask.append(mask)

        edge_mask = torch.cat(edge_mask, dim=0)
        edge_mask = torch.where(edge_mask, 0, -torch.inf)
        stop_mask = torch.where(stop_mask, 0, -torch.inf)
        return edge_mask, stop_mask

    # def make_edge_mask(self, g, states, cond_states):
    #     """
    #     edge-level mask (invalid actions get -torch.inf)
    #     """
    #     edge_mask = [
    #         torch.tensor([], dtype=torch.bool).reshape(0, self.env.num_edge_types)
    #     ]
    #     stop_mask = torch.empty(len(g.edgelv_loc), dtype=torch.bool)
    #     for i, k in enumerate(g.edgelv_loc):
    #         k = k.item()
    #         s, c = states[k], cond_states[k]

    #         tgt = sorted(s.edge_targets)
    #         cond_tgt = s.edge_targets & set(c.graph[s.edge_source])

    #         e_i = torch.LongTensor([tgt.index(i) for i in cond_tgt])
    #         e_mask = index_utils.to_mask(e_i, len(tgt))

    #         et_i = torch.LongTensor(
    #             [c.graph.edges[s.edge_source, i]["edge_type"] for i in cond_tgt]
    #         )
    #         row_mask = e_mask[:, None].repeat(1, self.env.num_edge_types)
    #         src = torch.ones(len(et_i), 1, dtype=torch.bool)
    #         col_mask = torch.zeros_like(row_mask).scatter_(
    #             dim=1, index=et_i.unsqueeze(1), src=src
    #         )

    #         edge_mask.append(row_mask * col_mask)
    #         stop_mask[i] = False if cond_tgt else True

    #     edge_mask = torch.cat(edge_mask, dim=0)
    #     edge_mask = torch.where(edge_mask, 0, -torch.inf)
    #     stop_mask = torch.where(stop_mask, 0, -torch.inf)
    #     return edge_mask, stop_mask

    @property
    def device(self):
        return next(self.parameters()).device
