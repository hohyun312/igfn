import numpy as np
import torch

from torch.distributions import Categorical
from torch_scatter import scatter_max, scatter_log_softmax

from src.containers import GraphAction, GraphActionType
import src.index_utils as index_utils


def classify_actions_by_state_types(actions):
    init_actions = [a for a in actions if a.action_type == GraphActionType.Init]
    nodelv_actions = [
        a
        for a in actions
        if a.action_type in [GraphActionType.AddNode, GraphActionType.StopNode]
    ]
    edgelv_actions = [
        a
        for a in actions
        if a.action_type in [GraphActionType.AddEdge, GraphActionType.StopEdge]
    ]
    return dict(
        init_actions=init_actions,
        nodelv_actions=nodelv_actions,
        edgelv_actions=edgelv_actions,
    )


class DefaultCategorical(Categorical):
    def __init__(self, logits):
        """
        Set default value when `logits` is empty.
        """
        self.logits = logits
        self._is_empty = len(logits) == 0
        self.size = len(logits)

        if not self._is_empty:
            super().__init__(logits=logits)

    def sample(self):
        if self._is_empty:
            return torch.empty_like(self.logits, dtype=torch.long).flatten()
        return super().sample()

    def log_prob(self, value):
        if self._is_empty:
            return torch.empty_like(self.logits, dtype=torch.float).flatten()
        return super().log_prob(value)


class FlatCategorical:
    def __init__(self, logits, indices):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert indices.dtype == torch.long, "`indices` should be torch.long data type"
        assert (
            logits.shape == indices.shape
        ), f"The shape of `logits` and `indices` should be the same, got logits={logits.shape}, indices={indices.shape}"

        indices = index_utils.relabel_mapping(indices)[indices]

        # we don't assume indices are non-decreasing.
        self.indices, self._si = torch.sort(indices, stable=True)
        self.logits = logits[self._si]
        self.size = max(self.indices.tolist(), default=-1) + 1

    def log_prob(self, value):
        assert self.size == len(
            value
        ), f"size doesn't match, size {self.size}, got {value.shape}"
        log_probs = scatter_log_softmax(self.logits, self.indices)
        return log_probs[value + index_utils.compute_offset(self.indices)]

    @torch.no_grad()
    def sample(self):
        unif = torch.rand_like(self.logits)
        gumbel = -(-unif.log()).log()
        _, samples = scatter_max(self.logits + gumbel, self.indices)
        return samples - index_utils.compute_offset(self.indices)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class ActionCategorical:
    def __init__(
        self,
        init_cat,
        nodelv_cat,
        edgelv_cat,
        num_edge_types,
        from_index,
        cond_info=None,
    ):
        self.init_cat = init_cat
        self.nodelv_cat = nodelv_cat
        self.edgelv_cat = edgelv_cat

        self._to_index = index_utils.transpose(from_index)
        self._num_edge_types = num_edge_types
        self._cond_info = cond_info

        # self.init_cat = FlatCategorical(
        #         logits=init_logits, indices=cond_info["init_indices"]
        #     )

        self.sizes = (self.init_cat.size, nodelv_cat.size, self.edgelv_cat.size)

    def log_prob(self, actions):
        table = classify_actions_by_state_types(actions)
        init_log_prob = self.init_logprob(table["init_actions"])
        node_log_prob = self.nodelv_logprob(table["nodelv_actions"])
        edge_log_prob = self.edgelv_logprob(table["edgelv_actions"])
        log_probs = torch.cat([init_log_prob, node_log_prob, edge_log_prob], 0)
        return log_probs[self._to_index]

    def sample(self):
        samples = (
            self.init_tensor_to_actions(self.init_cat.sample())
            + self.nodelv_tensor_to_actions(self.nodelv_cat.sample())
            + self.edgelv_tensor_to_actions(self.edgelv_cat.sample())
        )
        return [samples[i.item()] for i in self._to_index]

    def init_logprob(self, actions):
        value = self.init_actions_to_tensor(actions)
        return self.init_cat.log_prob(value)

    def nodelv_logprob(self, actions):
        value = self.nodelv_actions_to_tensor(actions)
        return self.nodelv_cat.log_prob(value)

    def edgelv_logprob(self, actions):
        value = self.edgelv_actions_to_tensor(actions)
        return self.edgelv_cat.log_prob(value)

    def init_actions_to_tensor(self, actions):
        if self._cond_info is None:
            return torch.LongTensor([a.node_type for a in actions])
        else:
            return torch.LongTensor([a.target for a in actions])

    def init_tensor_to_actions(self, tensor):
        if self._cond_info is None:
            return [
                GraphAction(GraphActionType.Init, node_type=i.item()) for i in tensor
            ]
        else:
            return [
                GraphAction(
                    GraphActionType.Init,
                    node_type=self._cond_info["node_types"][i].item(),
                    target=i.item(),
                    node_label=self._cond_info["node_labels"][i].item(),  # TODO
                )
                for i in tensor
            ]

    def nodelv_actions_to_tensor(self, actions):
        return torch.LongTensor(
            [
                0
                if a.action_type == GraphActionType.StopNode
                else (1 + self._num_edge_types * a.node_type + a.edge_type)
                for a in actions
            ]
        )

    def nodelv_tensor_to_actions(self, tensor):
        # stop, nodetype=0, nodetype=0, nodetype=1, nodetype=1, ...
        #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = GraphAction(GraphActionType.StopNode)
            else:
                a = GraphAction(
                    GraphActionType.AddNode,
                    node_type=(i - 1) // self._num_edge_types,
                    edge_type=(i - 1) % self._num_edge_types,
                )
            actions.append(a)
        return actions

    def edgelv_actions_to_tensor(self, actions):
        return torch.LongTensor(
            [
                0
                if a.action_type == GraphActionType.StopEdge
                else (1 + self._num_edge_types * a.target + a.edge_type)
                for a in actions
            ]
        )

    def edgelv_tensor_to_actions(self, tensor):
        # stop,   target=0,   target=0,   target=1,   target=1, ...
        #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = GraphAction(GraphActionType.StopEdge)
            else:
                a = GraphAction(
                    GraphActionType.AddEdge,
                    target=(i - 1) // self._num_edge_types,
                    edge_type=(i - 1) % self._num_edge_types,
                )
            actions.append(a)
        return actions

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes: {self.sizes})"


class CondActionCategorical:
    def __init__(
        self,
        init_cat,
        nodelv_cat,
        edgelv_cat,
        num_edge_types,
        from_index,
        cond_info=None,
    ):
        self.init_cat = init_cat
        self.nodelv_cat = nodelv_cat
        self.edgelv_cat = edgelv_cat

        self._to_index = index_utils.transpose(from_index)
        self._num_edge_types = num_edge_types
        self._cond_info = cond_info

        self.sizes = (self.init_cat.size, nodelv_cat.size, self.edgelv_cat.size)

    def log_prob(self, actions):
        table = classify_actions_by_state_types(actions)
        init_log_prob = self.init_logprob(table["init_actions"])
        node_log_prob = self.nodelv_logprob(table["nodelv_actions"])
        edge_log_prob = self.edgelv_logprob(table["edgelv_actions"])
        log_probs = torch.cat([init_log_prob, node_log_prob, edge_log_prob], 0)
        return log_probs[self._to_index]

    def sample(self):
        samples = (
            self.init_tensor_to_actions(self.init_cat.sample())
            + self.nodelv_tensor_to_actions(self.nodelv_cat.sample())
            + self.edgelv_tensor_to_actions(self.edgelv_cat.sample())
        )
        return [samples[i.item()] for i in self._to_index]

    def init_logprob(self, actions):
        value = self.init_actions_to_tensor(actions)
        return self.init_cat.log_prob(value)

    def nodelv_logprob(self, actions):
        value = self.nodelv_actions_to_tensor(actions)
        return self.nodelv_cat.log_prob(value)

    def edgelv_logprob(self, actions):
        value = self.edgelv_actions_to_tensor(actions)
        return self.edgelv_cat.log_prob(value)

    def init_actions_to_tensor(self, actions):
        # TODO: self.init_mappings
        return torch.LongTensor(
            [m[a.node_label] for m, a in zip(self.init_mappings, actions)]
        )

    def init_tensor_to_actions(self, tensor):
        # TODO: self.init_node_types
        return [
            GraphAction(
                GraphActionType.Init,
                node_type=nt[i.item()],
                node_label=i.item(),
            )
            for nt, i in zip(self.init_node_types, tensor)
        ]

    def nodelv_actions_to_tensor(self, actions):
        # TODO: self.node_mappings
        return torch.LongTensor(
            [m[a.node_label] for m, a in zip(self.node_mappings, actions)]
        )

    def nodelv_tensor_to_actions(self, tensor):
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = GraphAction(GraphActionType.StopNode)
            else:
                j = i - 1
                a = GraphAction(
                    GraphActionType.AddNode,
                    node_type=self.nodelv_node_types[j],
                    edge_type=self.nodelv_edge_types[j],
                    node_label=self.nodelv_node_labels[j],
                )
            actions.append(a)
        return actions

    def edgelv_actions_to_tensor(self, actions):
        return torch.LongTensor(
            [
                0
                if a.action_type == GraphActionType.StopEdge
                else (1 + self._num_edge_types * a.target + a.edge_type)
                for a in actions
            ]
        )

    def edgelv_tensor_to_actions(self, tensor):
        # stop,   target=0,   target=0,   target=1,   target=1, ...
        #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = GraphAction(GraphActionType.StopEdge)
            else:
                a = GraphAction(
                    GraphActionType.AddEdge,
                    target=(i - 1) // self._num_edge_types,
                    edge_type=(i - 1) % self._num_edge_types,
                )
            actions.append(a)
        return actions

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes: {self.sizes})"


if __name__ == "__main__":
    from unittest import TestCase

    class Test(TestCase):
        def test_flatcat_log_prob(self):
            logits = torch.tensor([1, 1, 1, 1, 1, 1]).float()
            indices = torch.tensor([0, 0, 1, 0, 1, 2])
            value = torch.tensor([1, 1, 0])

            log_probs = scatter_log_softmax(logits, indices)
            log_probs_ = FlatCategorical(logits, indices).log_prob(value)

            value = all(log_probs[[1, -2, -1]] == log_probs_)
            self.assertTrue(value)
