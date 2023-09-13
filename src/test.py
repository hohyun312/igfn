import numpy as np
import torch
import networkx as nx

from src.random_graph import n_community
from src.containers import (
    GraphStateType,
    GraphState,
    GraphActionType,
    GraphAction,
)
from src.graph_env import GraphEnv
from src.models import GraphModel


from unittest import TestCase


def wl_iso_mol():
    # decalin
    # C1CCC2CCCCC2C1
    # Bicyclopentyl
    # C1CCC(C1)C2CCCC2
    from src.mgraph import MolGraph

    states = [
        GraphState(GraphStateType.Terminal, MolGraph.from_smiles("C1CCC2CCCCC2C1")),
        GraphState(GraphStateType.Terminal, MolGraph.from_smiles("C1CCC(C1)C2CCCC2")),
    ]
    return states


def fake_data():
    states = [GraphState.initial_state()]
    actions = [
        GraphAction(GraphActionType.Init, node_type=0, target=0),
        GraphAction(GraphActionType.AddNode, node_type=0, edge_type=0),
        GraphAction(GraphActionType.AddNode, node_type=0, edge_type=1),
        GraphAction(GraphActionType.AddEdge, edge_type=0, target=0),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.AddNode, node_type=1, edge_type=0),
        GraphAction(GraphActionType.AddNode, node_type=1, edge_type=0),
        GraphAction(GraphActionType.StopEdge),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.StopNode),
    ]
    actions = [
        GraphAction(GraphActionType.Init, node_type=0, target=0),
        GraphAction(GraphActionType.AddNode, node_type=0, edge_type=0, target=1),
        GraphAction(GraphActionType.AddNode, node_type=0, edge_type=0, target=2),
        GraphAction(GraphActionType.StopEdge, edge_type=0, target=0),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.StopNode),
        GraphAction(GraphActionType.StopNode),
    ]
    for act in actions:
        states += [states[-1].apply_action(act)]
    return states, actions, states[-1]


c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], 2)
G = n_community([5, 7], p=0.4, p_inter=0.1)

env = GraphEnv(2, 2)


def follow_actions(env, traj):
    s = env.initial_state()
    for a in traj.actions:
        s = env.step(s, a)
    return s


def train_overfit(model, traj, n_updates=200):
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(n_updates):
        cat = model(traj.states[:-1])
        log_probs = cat.log_prob(traj.actions)
        loss = -log_probs.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def sample_from_model(env, model):
    state = env.initial_state()
    for _ in range(300):
        cat = model([state])
        action = cat.sample()[0]
        state = env.step(state, action)

        if state.state_type == env.GraphStateType.Terminal:
            break
    return state


class Test(TestCase):
    def test_trajectory1(self):
        traj = env.graph_bfs_trajectory(G, new_index=True)
        new_traj = env.follow_actions(traj.actions)
        is_iso = nx.is_isomorphic(G, new_traj.last_state.graph)
        self.assertTrue(is_iso)

    def test_trajectory2(self):
        traj = env.graph_bfs_trajectory(G, new_index=False)
        new_traj = env.follow_actions(traj.actions)
        is_iso = nx.is_isomorphic(G, new_traj.last_state.graph)
        self.assertTrue(is_iso)

    def test_overfit1(self):
        traj = env.graph_bfs_trajectory(G, new_index=True)
        model = GraphModel(env)
        train_overfit(model, traj)
        state = sample_from_model(env, model)
        is_iso = nx.is_isomorphic(state.graph, G)
        self.assertTrue(is_iso)

    def test_overfit2(self):
        traj = env.graph_bfs_trajectory(G, new_index=False)
        model = GraphModel(env)
        train_overfit(model, traj)
        state = sample_from_model(env, model)
        is_iso = nx.is_isomorphic(state.graph, G)
        self.assertTrue(is_iso)
