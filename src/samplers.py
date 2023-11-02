from typing import Optional
from copy import deepcopy
from collections import deque
from functools import partial


import numpy as np
import torch
from torch.distributions import Categorical
from torch_scatter import scatter_log_softmax, scatter_max


from src.containers import (
    State,
    StateType,
    BatchedState,
    Action,
    ActionType,
    Trajectory,
    BatchedTrajectory,
)


class Sampler:
    def __init__(self, model, max_num_nodes=40, max_step=100, max_degree=4):
        self.model = model
        self.max_num_nodes = max_num_nodes
        self.max_step = max_step
        self.max_degree = max_degree

    def initial_state(self):
        return State(StateType.Initial)

    def terminate(self, state: State):
        state.state_type = StateType.Terminal
        state.node_source = None
        state.edge_source = None
        state.target_range = None
        return state

    def step(self, state: State, action: Action, copy: bool = True):
        if copy:
            state = deepcopy(state)
        else:
            state = state

        if action.action_type == ActionType.First:
            assert state.state_type == StateType.Initial
            # Initial -> NodeLevel
            state.node_source = state.num_nodes
            state.state_type = StateType.NodeLevel
            state.add_node(action.node_type)

        elif action.action_type == ActionType.AddNode:
            # NodeLevel -> EdgeLevel
            state.edge_source = state.num_nodes
            state.state_type = StateType.EdgeLevel
            state.target_range = (state.node_source + 1, state.edge_source)
            state.add_node(action.node_type)
            state.add_edge(state.node_source, state.edge_source, action.edge_type)

            if state.target_range[0] == state.target_range[1]:
                action = Action(ActionType.StopEdge)
                state = self.step(state, action, copy=False)

        elif action.action_type == ActionType.AddEdge:
            # EdgeLevel -> EdgeLevel
            target = state.target_range[0] + action.target
            state.target_range = (target + 1, state.target_range[1])
            state.add_edge(state.edge_source, target, action.edge_type)

            if state.target_range[0] == state.target_range[1]:
                action = Action(ActionType.StopEdge)
                state = self.step(state, action, copy=False)

        elif action.action_type == ActionType.StopEdge:
            # EdgeLevel -> NodeLevel
            state.state_type = StateType.NodeLevel
            state.edge_source = None
            state.target_range = None

            degree = state.degree[state.node_source]
            if degree >= self.max_degree:
                action = Action(ActionType.StopNode)
                state = self.step(state, action, copy=False)

        elif action.action_type == ActionType.StopNode:
            # NodeLevel -> NodeLevel or Terminal
            if state.node_source + 1 < state.num_nodes:
                state.node_source += 1
            else:
                state.state_type = StateType.Terminal
                state.node_source = None
                state.target_range = None

        return state

    @torch.no_grad()
    def sample_terminal(self, size=1):
        is_training = self.model.training
        self.model.eval()
        batch = BatchedState([self.initial_state() for _ in range(size)])
        state_idx = list(range(size))
        done_state = [None] * size

        for _ in range(self.max_step):
            action = self.model.forward_action(batch).sample()
            next_batch, next_state_idx = [], []

            for i in range(len(batch)):
                new = self.step(batch[i], action[i])
                sid = state_idx[i]
                if new.state_type == StateType.Terminal:
                    done_state[sid] = new
                elif new.num_nodes >= self.max_num_nodes:
                    done_state[sid] = self.terminate(new)
                else:
                    next_batch.append(new)
                    next_state_idx.append(sid)

            batch = BatchedState(next_batch)
            state_idx = next_state_idx
            if batch.total_size == 0:
                break

        for i in range(len(batch)):
            state, sid = batch[i], state_idx[i]
            done_state[sid] = self.terminate(state)

        self.model.train(is_training)
        return BatchedState(done_state)

    def to_trajectory(self, input_state: State, node_order: list = None):
        assert input_state.state_type == StateType.Terminal
        if node_order is None:
            assert input_state.node_order, "node_order must be given"
            node_order = input_state.node_order

        rank = np.argsort(node_order)  # order_index
        input_adj = [
            sorted(input_state.adj[u], key=lambda x: rank[x])
            for u in range(input_state.num_nodes)
        ]

        root = node_order[0]
        in2out = [-1] * input_state.num_nodes
        in2out[root] = 0
        out2in = [-1] * input_state.num_nodes
        out2in[0] = root

        state = self.initial_state()
        action = Action(ActionType.First, node_type=input_state.node_type[root])

        traj = Trajectory()
        traj.add_state(state)
        state = self.step(state, action)
        traj.add_state_and_action(state, action)

        not_done = True
        while not_done:
            if state.state_type == StateType.NodeLevel:
                u = out2in[state.node_source]
                u_neighbors = input_adj[u]

                flg = False
                for v in u_neighbors:
                    if in2out[v] == -1:
                        flg = True
                        break

                if flg:
                    in2out[v] = state.num_nodes
                    out2in[state.num_nodes] = v
                    action = Action(
                        ActionType.AddNode,
                        node_type=input_state.node_type[v],
                        edge_type=input_state.e2t[(u, v)],
                    )
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)
                else:
                    action = Action(ActionType.StopNode)
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)

            elif state.state_type == StateType.EdgeLevel:
                v = out2in[state.edge_source]
                flg = False
                for i in range(*state.target_range):
                    t = out2in[i]
                    if (t, v) in input_state.e2t:
                        flg = True
                        break

                if flg:
                    target_node = in2out[t]
                    target = target_node - state.target_range[0]
                    action = Action(
                        ActionType.AddEdge,
                        target=target,
                        edge_type=input_state.e2t[(t, v)],
                    )
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)
                else:
                    action = Action(ActionType.StopEdge)
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)

            elif state.state_type == StateType.Terminal:
                not_done = False
        return traj

    @torch.no_grad()
    def to_batched_trajectory(
        self, batch: BatchedState, node_order: Optional[torch.Tensor] = None
    ):
        if node_order is None:
            node_order = self.model.backward_action(batch).sample()
        split_size = torch.Size(batch.num_nodes())
        node_order_split = node_order.cpu().split(split_size)
        traj = BatchedTrajectory()
        for i in range(batch.total_size):
            node_order_lst = node_order_split[i].tolist()
            traj_i = self.to_trajectory(batch[i], node_order_lst)
            traj.add(traj_i)
        return traj


class ForwardActionDistribution:
    def __init__(
        self,
        init_logits,
        nodelv_logits,
        edgelv_logits,
        edgelv_index,
        num_edge_types,
        from_index,
    ):
        self.init_cat = DefaultCategorical(logits=init_logits)
        self.nodelv_cat = DefaultCategorical(logits=nodelv_logits)
        self.edgelv_cat = VariadicCategorical(logits=edgelv_logits, indices=edgelv_index)
        self.num_edge_types = num_edge_types

        self._to_index = torch.argsort(from_index)

        self.sizes = (self.init_cat.size, self.nodelv_cat.size, self.edgelv_cat.size)
        self.device = init_logits.device

    def _classify_actions_by_types(self, actions):
        init_actions = [a for a in actions if a.action_type == ActionType.First]
        nodelv_actions = [
            a
            for a in actions
            if a.action_type in {ActionType.AddNode, ActionType.StopNode}
        ]
        edgelv_actions = [
            a
            for a in actions
            if a.action_type in {ActionType.AddEdge, ActionType.StopEdge}
        ]
        return dict(
            init_actions=init_actions,
            nodelv_actions=nodelv_actions,
            edgelv_actions=edgelv_actions,
        )

    def log_prob(self, actions):
        m = self._classify_actions_by_types(actions)
        init_log_prob = self.init_logprob(m["init_actions"])
        node_log_prob = self.nodelv_logprob(m["nodelv_actions"])
        edge_log_prob = self.edgelv_logprob(m["edgelv_actions"])
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
        return torch.tensor([a.node_type for a in actions], dtype=torch.long, device=self.device)

    def init_tensor_to_actions(self, tensor):
        return [Action(ActionType.First, node_type=i.item()) for i in tensor]

    def nodelv_actions_to_tensor(self, actions):
        return torch.tensor(
            [
                0
                if a.action_type == ActionType.StopNode
                else (1 + self.num_edge_types * a.node_type + a.edge_type)
                for a in actions
            ], dtype=torch.long, device=self.device
        )

    def nodelv_tensor_to_actions(self, tensor):
        # stop, nodetype=0, nodetype=0, nodetype=1, nodetype=1, ...
        #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = Action(ActionType.StopNode)
            else:
                a = Action(
                    ActionType.AddNode,
                    node_type=(i - 1) // self.num_edge_types,
                    edge_type=(i - 1) % self.num_edge_types,
                )
            actions.append(a)
        return actions

    def edgelv_actions_to_tensor(self, actions):
        return torch.tensor(
            [
                0
                if a.action_type == ActionType.StopEdge
                else (1 + self.num_edge_types * a.target + a.edge_type)
                for a in actions
            ], dtype=torch.long, device=self.device
        )

    def edgelv_tensor_to_actions(self, tensor):
        # stop,   target=0,   target=0,   target=1,   target=1, ...
        #     , edgetype=0, edgetype=1, edgetype=0, edgetype=1, ...
        actions = []
        for i in tensor:
            i = i.item()
            if i == 0:
                a = Action(ActionType.StopEdge)
            else:
                a = Action(
                    ActionType.AddEdge,
                    target=(i - 1) // self.num_edge_types,
                    edge_type=(i - 1) % self.num_edge_types,
                )
            actions.append(a)
        return actions

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes: {self.sizes})"


class BackwardActionDistribution:
    def __init__(self, logits, sizes, neighbors, multiplicity):
        """
        Plackett Luce model for ranking data.
        This class also considers sampling order that conforms to BFS ordering
        """
        self.logits = logits - logits.max()
        self.sizes = torch.Size(sizes)
        self.neighbors = neighbors
        self.multiplicity = multiplicity
        self.unprob = torch.exp(self.logits).split(
            self.sizes
        )  # unnormalized probability

    def log_prob(self, node_order):
        splits = node_order.split(self.sizes)
        logp = [self.log_prob_i(i, o.tolist()) for i, o in enumerate(splits)]
        return torch.stack(logp) + torch.log(self.multiplicity)

    def sample(self, return_logp=False):
        if return_logp:
            samples, logp = zip(
                *[self.sample_i(i, True) for i in range(len(self.sizes))]
            )
            return torch.cat(samples), torch.stack(logp)

        else:
            return torch.cat([self.sample_i(i) for i in range(len(self.sizes))])

    def log_prob_i(self, i, node_order):
        logp = torch.log(self.unprob[i]) - torch.log(self.unprob[i].sum())
        root = node_order[0]
        total_logp = logp[root]
        node_rank = np.argsort(node_order)

        siblings = self._bfs_siblings(
            i, root, partial(self._given_order, node_rank=node_rank)
        )
        for sib in siblings:
            if len(sib) <= 1:
                continue
            unprob = self.unprob[i][sib[::-1]]
            unlogp = unprob.log().sum()
            logz = torch.cumsum(unprob, 0).log().sum()
            total_logp = total_logp + unlogp - logz
        return total_logp

    def sample_i(self, i, return_logp=False):
        prob = self.unprob[i].cpu().detach().numpy()
        root = np.random.choice(np.arange(len(prob)), p=prob / prob.sum())
        output = [root]

        siblings = self._bfs_siblings(i, root, partial(self._sample_order, prob=prob))

        total_logp = 0
        for sib in siblings:
            output.extend(sib)

            if len(sib) > 1 and return_logp:
                unprob = self.unprob[i][sib[::-1]]
                unlogp = unprob.log().sum()
                logz = torch.cumsum(unprob, 0).log().sum()
                total_logp = total_logp + unlogp - logz

        output = torch.LongTensor(output)

        if return_logp:
            logp = torch.log(self.unprob[i]) - torch.log(self.unprob[i].sum())
            total_logp = total_logp + logp[root]
            return output, total_logp

        return output

    def _sample_order(self, neighbors, prob):
        order = self._plackett_luce_sample(prob[neighbors])
        neighbors = [neighbors[i] for i in order]
        return neighbors

    def _given_order(self, neighbors, node_rank):
        return sorted(neighbors, key=lambda x: node_rank[x])

    def _bfs_siblings(self, i, root, visit_strategy):
        neighbors_i = self.neighbors[i]
        parents = deque([root])
        visited = set([int(root)])
        while parents:
            p = parents.popleft()
            children = []
            p_neighbors = visit_strategy(neighbors_i[p])
            for nbr in p_neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    parents.append(nbr)
                    children.append(nbr)
            yield children

    def _plackett_luce_sample(self, p: np.ndarray):
        indices = np.arange(len(p))
        output = []
        for _ in range(len(p)):
            i = np.random.choice(np.arange(len(p)), p=p / p.sum())
            output.append(indices[i])
            mask = np.ones(len(p), dtype=bool)
            mask[i] = False
            indices = indices[mask]
            p = p[mask]
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes={self.sizes})"


class DefaultCategorical(Categorical):
    def __init__(self, logits):
        """
        Return the default value when `logits` is empty tensor.
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


def _rank_values(indices):
    """
    input : [3, 3, 4, 7, 7, 0]
    output: [1, 1, 2, 3, 3, 0]
    """
    u = indices.unique(sorted=True)
    s = torch.arange(len(u), device=indices.device)
    mapping = torch.full(
        (max(s, default=0) + 1,), 
        -1, 
        dtype=torch.long, 
        device=indices.device
    ).scatter_(
        dim=0, index=u, src=s
    )
    return mapping[indices]


def _batch_offset(indices):
    """
    compute offset that is useful for batch operation

    input : [0, 0, 0, 1, 2, 2]
    output: [0, 3, 4]
    """
    count = torch.bincount(indices)
    return torch.cumsum(count, 0) - count


class VariadicCategorical:
    def __init__(self, logits, indices):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert indices.dtype == torch.long, "`indices` should be torch.long data type"
        assert (
            logits.shape == indices.shape
        ), f"The shape of `logits` and `indices` should be the same, got logits={logits.shape}, indices={indices.shape}"

        batch = _rank_values(indices)

        # we don't assume indices are non-decreasing.
        self.batch, _sort_indices = torch.sort(batch, stable=True)
        self.logits = logits[_sort_indices]
        self._offset = _batch_offset(self.batch)
        self.size = max(self.batch.tolist(), default=-1) + 1

    def log_prob(self, value):
        assert self.size == len(
            value
        ), f"size doesn't match, size {self.size}, got {value.shape}"
        log_probs = scatter_log_softmax(self.logits, self.batch)
        return log_probs[value + self._offset]

    @torch.no_grad()
    def sample(self, eps=1e-16):
        """
        Sample from softmax probabilities using Gumbel-max trick.
        """
        unif = torch.rand_like(self.logits) + eps
        gumbel = -(-unif.log() + eps).log()
        _, sample_ids = scatter_max(self.logits + gumbel, self.batch)
        return sample_ids - self._offset

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
