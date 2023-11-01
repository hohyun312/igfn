from unittest import TestCase

from collections import Counter
import math

import networkx as nx

import torch
from torch_scatter import scatter_log_softmax

from src.samplers import VariadicCategorical, BackwardActionDistribution
from src.types import (
    GraphBuildingEnv,
    State,
    StateType,
    Action,
    ActionType,
    Trajectory,
    BatchedState,
)
from src.models import GraphPolicy, GraphEmbedding


regular_state = State(  # d=3, n=10, regular graph
    StateType.Terminal,
    [0] * 10,
    [0] * 30,
    [
        [0, 6],
        [6, 0],
        [0, 9],
        [9, 0],
        [2, 4],
        [4, 2],
        [2, 6],
        [6, 2],
        [2, 9],
        [9, 2],
        [3, 1],
        [1, 3],
        [3, 5],
        [5, 3],
        [3, 8],
        [8, 3],
        [4, 0],
        [0, 4],
        [4, 8],
        [8, 4],
        [5, 7],
        [7, 5],
        [6, 1],
        [1, 6],
        [7, 1],
        [1, 7],
        [8, 7],
        [7, 8],
        [9, 5],
        [5, 9],
    ],
)

num_node_types = 2
num_edge_types = 2

actions1 = [
    Action(ActionType.First, node_type=0),
    Action(ActionType.AddNode, node_type=0, edge_type=0),
    Action(ActionType.AddNode, node_type=0, edge_type=1),
    Action(ActionType.AddEdge, edge_type=0, target=0),
    Action(ActionType.StopNode),
    Action(ActionType.StopNode),
    Action(ActionType.AddNode, node_type=1, edge_type=0),
    Action(ActionType.AddNode, node_type=1, edge_type=0),
    Action(ActionType.StopEdge),
    Action(ActionType.StopNode),
    Action(ActionType.StopNode),
    Action(ActionType.StopNode),
]

actions2 = [
    Action(ActionType.First, node_type=0),
    Action(ActionType.AddNode, node_type=0, edge_type=0, target=1),
    Action(ActionType.AddNode, node_type=0, edge_type=0, target=2),
    Action(ActionType.StopEdge, edge_type=0, target=0),
    Action(ActionType.StopNode),
    Action(ActionType.StopNode),
    Action(ActionType.StopNode),
]

env = GraphBuildingEnv(3, 3, max_degree=10)


def follow_actions(actions):
    state = env.initial_state()
    states = [state]
    for action in actions:
        state = env.step(state, action, copy=True)
        states.append(state)
    return Trajectory(states, actions)


class TestEnv(TestCase):
    def test_step1(self):
        state = env.initial_state()
        for action in actions1:
            env.step(state, action, copy=False)
            self.check_valid_graph_state(state)

    def test_step2(self):
        state = env.initial_state()
        for action in actions2:
            env.step(state, action, copy=False)
            self.check_valid_graph_state(state)

    def check_valid_graph_state(self, state: State):
        # not exhaustive
        if state.state_type == StateType.EdgeLevel:
            self.assertIsNotNone(state.edge_source)
            self.assertIsNotNone(state.node_source)
        elif state.state_type == StateType.NodeLevel:
            self.assertIsNotNone(state.node_source)


class TestVarCat(TestCase):
    def test_log_prob(self):
        logits = torch.tensor([1, 1, 1, 1, 1, 1]).float()
        indices = torch.tensor([0, 0, 1, 0, 1, 2])
        value = torch.tensor([1, 1, 0])

        log_probs = scatter_log_softmax(logits, indices)
        log_probs_ = VariadicCategorical(logits, indices).log_prob(value)

        value = all(log_probs[[1, -2, -1]] == log_probs_)
        self.assertTrue(value)


def train_overfit(model, traj, n_updates=50):
    optimizer = torch.optim.Adam(model.parameters())
    losses = []
    for _ in range(n_updates):
        cat = model.forward_action(BatchedState(traj.get_states()))
        log_probs = cat.log_prob(traj.get_actions())
        loss = -log_probs.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def sample_from_model(env, model):
    state = env.initial_state()
    actions = []
    states = [state]
    for _ in range(300):
        cat = model.forward_action(BatchedState([state]))
        action = cat.sample()[0]
        state = env.step(state, action)
        states.append(state)
        actions.append(action)

        if state.state_type == StateType.Terminal:
            break
    return Trajectory(states, actions)


class TestModel(TestCase):
    def test_forward1(self):
        model = GraphPolicy(num_node_types, num_edge_types)
        traj = follow_actions(actions1)
        train_overfit(model, traj)
        sample = sample_from_model(env, model)
        G1, G2 = traj.get_last_state().to_nx(), sample.get_last_state().to_nx()

        self.assertTrue(nx.is_isomorphic(G1, G2))

    def test_forward2(self):
        model = GraphPolicy(num_node_types, num_edge_types)
        traj = follow_actions(actions2)
        train_overfit(model, traj)
        sample = sample_from_model(env, model)
        G1, G2 = traj.get_last_state().to_nx(), sample.get_last_state().to_nx()

        self.assertTrue(nx.is_isomorphic(G1, G2))


class TestBFSPlackettLuce(TestCase):
    def test_logprob(self):
        logit = torch.ones(5)
        neighbors = [[[1, 2, 3, 4], [0], [0], [0], [0]]]
        sizes = torch.tensor([5])

        bfspl = BackwardActionDistribution(logit, sizes, neighbors, torch.tensor(1.0))

        node_order = torch.LongTensor([0, 1, 2, 3, 4])
        prob = bfspl.log_prob(node_order).exp()
        d = math.factorial(5)

        self.assertAlmostEqual(prob.item(), 1 / d)

        node_order = torch.LongTensor([1, 0, 2, 3, 4])
        prob = bfspl.log_prob(node_order).exp()

        self.assertAlmostEqual(prob.item(), 4 / d)

    def test_sample(self):
        logit = torch.ones(4)
        neighbors = [[[1, 2, 3], [0], [0], [0]]]
        sizes = torch.tensor([4])
        bfspl = BackwardActionDistribution(logit, sizes, neighbors, torch.tensor(1.0))

        N = 0
        for _ in range(1000):
            N += bfspl.sample()[0].item() == 0

        self.assertTrue(220 < N < 280)
