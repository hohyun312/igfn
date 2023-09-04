import numpy as np
import torch

from torch.distributions import Categorical
from torch_scatter import scatter_max, scatter_log_softmax

from src.graph_building_env import GraphAction, GraphActionType


def index_arange(indices):
    # E.g. indices: [3, 3, 4, 7, 7, 0]
    #       output: [1, 1, 2, 3, 3, 0]
    u = indices.unique(sorted=True)
    s = torch.arange(len(u))
    mapping = torch.full((u.max() + 1,), -1, dtype=torch.long).scatter_(
        dim=0, index=u, src=s
    )
    return mapping[indices]


class NullActionCategorical:
    def __init__(self):
        self.num_data = 0

    def sample(self):
        return []

    def log_prob(self, actions):
        return torch.tensor([])


class BaseActionCategorical:
    def __new__(cls, logits, *args, **kargs):
        if logits.numel() == 0:
            return NullActionCategorical()
        else:
            return super().__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_data: {self.num_data})"


class AuxInitActionCategorical(BaseActionCategorical):
    def __init__(self, logits, indices):
        self.batch = indices.unique(sorted=True)
        self.num_data = len(self.batch)
        self.logits = logits.flatten()
        self.indices = index_arange(indices)

    def sample(self):
        u = torch.rand_like(self.logits)
        gumbel = -(-u.log()).log()
        _, samples = scatter_max(self.logits + gumbel, self.indices)
        actions = []
        for i in samples.tolist():
            a = GraphAction(GraphActionType.Init, target=i)
            actions.append(a)
        return actions

    def log_prob(self, actions):
        targets = torch.LongTensor([a.target for a in actions])
        log_probs = scatter_log_softmax(self.logits, self.indices)
        return log_probs[self.batch + targets]


class InitActionCategorical(BaseActionCategorical):
    def __init__(self, logits):
        self.num_data = len(logits)
        self.logits = logits
        self.cat = Categorical(logits=logits)

    @torch.no_grad()
    def sample(self):
        samples = self.cat.sample()
        actions = [
            GraphAction(GraphActionType.Init, node_type=i) for i in samples.tolist()
        ]
        return actions

    def log_prob(self, actions):
        index = torch.LongTensor([a.node_type for a in actions])
        return self.cat.log_prob(index)


class NodeLevelActionCategorical(BaseActionCategorical):
    def __init__(self, logits, num_edge_types):
        self.num_data = len(logits)
        self.logits = logits
        self.num_edge_types = num_edge_types
        self.cat = Categorical(logits=logits)

    @torch.no_grad()
    def sample(self):
        samples = self.cat.sample()
        actions = []
        for i in samples.tolist():
            # stop, nodetype=0, nodetype=0, nodetype=1, nodetype=1, ...
            #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
            if i == 0:
                a = GraphAction(GraphActionType.StopNode)
            else:
                node_type = (i - 1) // self.num_edge_types
                edge_type = (i - 1) % self.num_edge_types
                a = GraphAction(
                    GraphActionType.AddNode,
                    node_type=node_type,
                    edge_type=edge_type,
                )

            actions.append(a)
        return actions

    def log_prob(self, actions):
        index = torch.LongTensor(
            [
                0
                if a.action_type == GraphActionType.StopNode
                else (1 + self.num_edge_types * a.node_type + a.edge_type)
                for a in actions
            ]
        )
        return self.cat.log_prob(index)


class EdgeLevelActionCategorical(BaseActionCategorical):
    def __init__(
        self, stop_logits, edge_logits, stop_edge_batch, non_edge_batch, num_edge_types
    ):
        self.num_data = len(stop_logits)
        self.num_edge_types = num_edge_types
        self.logits = torch.cat([stop_logits.flatten(), edge_logits.flatten()], 0)

        # We don't assume batch index starts from 0 and increases linearly.
        # E.g. indices: [3, 3, 4, 7, 7] -> self.batch: [0, 0, 1, 2, 2]
        non_edge_batch = non_edge_batch.repeat_interleave(num_edge_types)
        indices = torch.cat([stop_edge_batch, non_edge_batch], 0)
        self.batch = index_arange(indices)

    @torch.no_grad()
    def sample(self):
        u = torch.rand_like(self.logits)
        gumbel = -(-u.log()).log()
        _, samples = scatter_max(self.logits + gumbel, self.batch)
        # samples = samples[samples != len(self.logits)]
        actions = []
        for i in samples.tolist():
            # stop,   target=0,   target=0,   target=1,   target=1, ...
            #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
            if i == 0:
                a = GraphAction(GraphActionType.StopEdge)
            else:
                target = (i - 1) // self.num_edge_types
                edge_type = (i - 1) % self.num_edge_types
                a = GraphAction(
                    GraphActionType.AddEdge,
                    target=target,
                    edge_type=edge_type,
                )
            actions.append(a)
        return actions

    def log_prob(self, actions):
        assert self.num_data == len(actions)
        edge_index = torch.LongTensor(
            [
                0
                if a.action_type == GraphActionType.StopEdge
                else (1 + self.num_edge_types * a.target + a.edge_type)
                for a in actions
            ]
        )
        log_probs = scatter_log_softmax(self.logits, self.batch)
        mask = edge_index == 0
        edge_index[mask] = torch.arange(self.num_data)[mask]
        if self.num_data != len(
            self.logits
        ):  # only stop actions are available when self.num_data == len(self.logits)
            cnt = torch.bincount(self.batch[self.num_data :], minlength=self.num_data)
            ptr = torch.cumsum(cnt, dim=0) - cnt + self.num_data
            edge_index[~mask] = edge_index[~mask] + ptr[~mask] - 1
        return log_probs[edge_index]


class GraphActionCategorical:
    def __init__(self, init_cat, node_cat, edge_cat, order):
        self.init_cat = init_cat
        self.node_cat = node_cat
        self.edge_cat = edge_cat
        self.order = order

    def sample(self):
        samples = (
            self.init_cat.sample() + self.node_cat.sample() + self.edge_cat.sample()
        )
        return np.array(samples)[self.order.tolist()].tolist()

    def log_prob(self, actions):
        init_actions = [a for a in actions if a.action_type == GraphActionType.Init]
        node_actions = [
            a
            for a in actions
            if a.action_type in [GraphActionType.AddNode, GraphActionType.StopNode]
        ]
        edge_actions = [
            a
            for a in actions
            if a.action_type in [GraphActionType.AddEdge, GraphActionType.StopEdge]
        ]
        init_log_prob = self.init_cat.log_prob(init_actions)
        node_log_prob = self.node_cat.log_prob(node_actions)
        edge_log_prob = self.edge_cat.log_prob(edge_actions)
        log_probs = torch.cat([init_log_prob, node_log_prob, edge_log_prob], 0)
        return log_probs[self.order]

    def __repr__(self):
        attr = ", ".join(
            [str(cat.num_data) for cat in [self.init_cat, self.node_cat, self.edge_cat]]
        )
        return f"{self.__class__.__name__}(num_data: {attr})"
