import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import torch_geometric.nn as gnn
import torch_geometric.data as gd
from torch_scatter import scatter_add

from src.containers import StateType, BatchedState, BatchedTrajectory
from src.samplers import (
    ForwardActionDistribution,
    BackwardActionDistribution,
    DefaultCategorical,
    VariadicCategorical,
)


class FullyConnected(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation="leaky_relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = getattr(F, activation)

    def forward(self, input):
        output = input
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        return self.layers[-1](output)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        return self


class TwoStreamMLP(nn.Module):
    def __init__(self, head1_param, head2_param):
        super().__init__()
        self.head1 = FullyConnected(*head1_param)
        self.head2 = FullyConnected(*head2_param)

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
                        FullyConnected(emb_dim, [emb_dim * 4], emb_dim),
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


class GraphEmbedding(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        num_edge_types: int,
        emb_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.emb_dim = emb_dim
        self.transf = GraphTransformer(emb_dim, num_layers, num_heads)
        self.emb = nn.ModuleDict(
            {
                "node_order": gnn.PositionalEncoding(emb_dim),
                "node_type": nn.Embedding(num_node_types + 1, emb_dim),
                "node_state": nn.Embedding(
                    4, emb_dim, padding_idx=0
                ),  # node_source, queue, edge_source
                "edge_type": nn.Embedding(num_edge_types, emb_dim),
                "glob_node": nn.Embedding(len(StateType), emb_dim),
            }
        )

    def forward(self, batch, terminal=False):
        g = self.collate(batch)
        if not terminal:
            assert batch.size_by_type[-1] == 0, "Terminal state may cause errors"
            g.x = (
                self.emb["node_type"](g.node_type)
                + self.emb["node_order"](g.node_order)
                + self.emb["node_state"](g.node_state_id)
            )
            g.cond = self.emb["glob_node"](g.graph_state_id)
        else:
            assert (
                sum(batch.size_by_type[:-1]) == 0
            ), "Non-terminal state may cause errors"
            g.x = self.emb["node_type"](g.node_type)
            g.cond = self.emb["glob_node"](g.graph_state_id)

        g.edge_attr = self.emb["edge_type"](g.edge_type)
        g.emb = self.transf(g)
        return g

    @property
    def device(self):
        return next(self.parameters()).device

    def collate(self, batch: BatchedState):
        data_list = [s.to_tensor() for s in batch.sort_states()]
        graphs = gd.Batch.from_data_list(data_list)
        graphs.sptr = torch.cumsum(torch.LongTensor([0] + batch.size_by_type), dim=0)
        return graphs.to(self.device)


class GraphPolicy(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        num_edge_types: int,
        emb_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        self.graph_embedding = GraphEmbedding(
            num_node_types, num_edge_types, emb_dim, num_layers, num_heads
        )
        self.num_edge_types = num_edge_types
        self.mlp = nn.ModuleDict(
            {
                "init": FullyConnected(emb_dim, [emb_dim], num_node_types),
                "nodelv": FullyConnected(
                    emb_dim,
                    [emb_dim],
                    num_edge_types * num_node_types + 1,
                ),
                "edgelv": TwoStreamMLP(
                    [emb_dim, [emb_dim], 1],
                    [emb_dim, [emb_dim], num_edge_types],
                ),
                "backward": FullyConnected(emb_dim, [emb_dim], 1),
            }
        )
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def backward_action(self, batch: BatchedState):
        g = self.graph_embedding(batch, terminal=True)
        logits = self.mlp["backward"](g.emb["node_embeddings"]).flatten()
        node_sizes = g.ptr[1:] - g.ptr[:-1]
        adjacency_lists = batch.adjacency_lists()
        multiplicity = batch.count_automorphisms().to(self.device)
        return BackwardActionDistribution(
            logits=logits,
            sizes=node_sizes,
            neighbors=adjacency_lists,
            multiplicity=multiplicity,
        )

    def forward_action(self, batch: BatchedState):
        g = self.graph_embedding(batch)

        _, i, j, k, _ = g.sptr
        init_glob = g.emb["graph_embeddings"][:i]
        nodelv_glob = g.emb["graph_embeddings"][i:j]
        edgelv_glob = g.emb["graph_embeddings"][j:k]
        ne_emb = g.emb["non_edge_embeddings"]

        init_logits = self.mlp["init"](init_glob)
        nodelv_logits = self.mlp["nodelv"](nodelv_glob)
        edgelv_logits = self.mlp["edgelv"](edgelv_glob, ne_emb)

        edgelv_stop_index = torch.arange(k - j, device=self.device)
        edgelv_tgt_index = torch.repeat_interleave(
            g.num_non_edges[j:k] * self.num_edge_types
        )
        edgelv_index = torch.cat([edgelv_stop_index, edgelv_tgt_index])

        init_cat = DefaultCategorical(logits=init_logits)
        nodelv_cat = DefaultCategorical(logits=nodelv_logits)
        edgelv_cat = VariadicCategorical(logits=edgelv_logits, indices=edgelv_index)

        return ForwardActionDistribution(
            init_logits,
            nodelv_logits,
            edgelv_logits,
            edgelv_index,
            self.num_edge_types,
            batch.sort_indices(),
        )

    def loss(self, traj: BatchedTrajectory, logR: torch.Tensor):  # logZ
        fwd_act = self.forward_action(traj.get_states())
        log_pf_s = fwd_act.log_prob(traj.get_actions())
        t_idx = torch.repeat_interleave(traj.get_length().to(self.device))
        log_pf = scatter_add(log_pf_s, t_idx)

        last_state = traj.get_last_state()
        bwd_act = self.backward_action(last_state)
        log_pb = bwd_act.log_prob(last_state.get_node_order())

        loss = (self.logZ + log_pf - log_pb - logR).square()
        return loss

    @property
    def device(self):
        return next(self.parameters()).device


class RewardModel(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        num_edge_types: int,
        emb_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        self.graph_embedding = GraphEmbedding(
            num_node_types, num_edge_types, emb_dim, num_layers, num_heads
        )
        self.mlp = FullyConnected(emb_dim, [2 * emb_dim], 1)

    def forward(self, batch: BatchedState):
        """log reward"""
        g = self.graph_embedding(batch, terminal=True)
        return self.mlp(g.emb["graph_embeddings"]).flatten()

    def loss(self, positive_sample, negative_sample, alpha=0.1):
        pos_out = self(positive_sample)
        neg_out = self(negative_sample)

        loss = pos_out - neg_out
        reg = pos_out.square() + neg_out.square()

        return -loss + alpha * reg

    @property
    def device(self):
        return next(self.parameters()).device
