import numpy as np
import random
from functools import cached_property
from collections import deque
import enum
from copy import deepcopy

import networkx as nx

import torch
import torch_geometric.data as gd


class GraphActionType(enum.Enum):
    Init = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    StopEdge = enum.auto()
    StopNode = enum.auto()


class GraphAction:
    def __init__(
        self,
        action_type: GraphActionType,
        target=None,
        node_type=None,
        edge_type=None,
    ):
        self.action_type = action_type
        self.target = target  # AddEdge, Init(B)
        self.node_type = node_type  # Init(F), AddNode
        self.edge_type = edge_type  # AddEdge, AddNode

    def __repr__(self):
        attr = ", ".join(
            [
                n + "=" + str(getattr(self, n))
                for n in ("target", "node_type", "edge_type")
                if getattr(self, n) is not None
            ]
        )
        return f"{self.__class__.__name__}(action_type={self.action_type.name}{attr})"


class GraphStateType(enum.Enum):
    Initial = enum.auto()
    EdgeLevel = enum.auto()
    NodeLevel = enum.auto()
    Terminal = enum.auto()


class NodeStateType:
    Ancester = 0
    NodeSource = 1
    EdgeSource = 2
    Frontier = 3


class GraphState:
    def __init__(
        self,
        state_type: GraphStateType,
        graph: nx.Graph,
        node_source=None,
        edge_source=None,
        frontier=deque(),
    ):
        self.state_type = state_type
        self.graph = graph
        self.node_source = node_source  # EdgeLevel, NodeLevel
        self.edge_source = edge_source  # EdgeLevel
        self.frontier = frontier

    def __repr__(self):
        return f"{self.__class__.__name__}(state_type={self.state_type.name}, num_nodes={len(self.graph)}, num_edges={len(self.graph.edges)})"

    @cached_property
    def edge_targets(self):
        if self.edge_source is None:
            return []
        return sorted(set(self.frontier) - set(self.graph[self.edge_source]))

    def cond_edge_targets(self, cond_state):
        if self.edge_source is None:
            return set()
        return set(cond_state.graph[self.edge_source])

    def cond_node_targets(self, cond_state):
        if self.node_source is None:
            return []
        inward_adj = self.graph[self.node_source]
        outward_adj = cond_state.graph[self.node_source]
        targets = set(outward_adj) - set(inward_adj)
        n, e = cond_state.graph.nodes, outward_adj
        return set([(n[t]["node_type"], e[t]["edge_type"]) for t in targets])

    @classmethod
    def initial_state(cls):
        return cls(GraphStateType.Initial, nx.Graph())

    def apply_action(self, action, copy=True):
        if copy:
            state = deepcopy(self)
        else:
            state = self

        if action.action_type == GraphActionType.Init:
            state.graph.add_node(0, node_type=action.node_type)
            state = GraphState(
                GraphStateType.NodeLevel, graph=state.graph, node_source=0
            )

        elif action.action_type == GraphActionType.AddNode:
            target = len(state.graph)
            state_type, edge_source = GraphStateType.EdgeLevel, target

            state.graph.add_node(target, node_type=action.node_type)
            state.graph.add_edge(state.node_source, target, edge_type=action.edge_type)
            state = GraphState(
                state_type,
                graph=state.graph,
                node_source=state.node_source,
                edge_source=edge_source,
                frontier=state.frontier,
            )
            if len(state.frontier) == 0:
                action = GraphAction(GraphActionType.StopEdge)
                state = state.apply_action(action, copy=False)

        elif action.action_type == GraphActionType.AddEdge:
            state.graph.add_edge(
                state.edge_source,
                state.edge_targets[action.target],
                edge_type=action.edge_type,
            )
            state = GraphState(
                GraphStateType.EdgeLevel,
                graph=state.graph,
                node_source=state.node_source,
                edge_source=state.edge_source,
                frontier=state.frontier,
            )
            if len(state.edge_targets) == 0:
                action = GraphAction(GraphActionType.StopEdge)
                state = state.apply_action(action, copy=False)

        elif action.action_type == GraphActionType.StopEdge:
            state.frontier.append(state.edge_source)
            state = GraphState(
                GraphStateType.NodeLevel,
                graph=state.graph,
                node_source=state.node_source,
                edge_source=None,
                frontier=state.frontier,
            )

        elif action.action_type == GraphActionType.StopNode:
            if len(state.frontier) > 0:
                node_source = state.frontier.popleft()
                state = GraphState(
                    GraphStateType.NodeLevel,
                    graph=state.graph,
                    node_source=node_source,
                    edge_source=None,
                    frontier=state.frontier,
                )
            else:
                state = GraphState(GraphStateType.Terminal, graph=state.graph)

        return state

    def to_tensor_graph(self):
        num_nodes = len(self.graph.nodes())
        node_type = torch.tensor(
            [attr["node_type"] for _, attr in self.graph.nodes(data=True)],
            dtype=torch.long,
        )
        edge_index = (
            torch.tensor(
                [e for i, j in self.graph.edges for e in [(i, j), (j, i)]],
                dtype=torch.long,
            )
            .reshape(-1, 2)
            .T
        )
        edge_type = torch.tensor(
            [
                (attr["edge_type"], attr["edge_type"])
                for _, _, attr in self.graph.edges(data=True)
            ],
            dtype=torch.long,
        ).flatten()
        non_edge_index = (
            torch.tensor(
                [(self.edge_source, v) for v in self.edge_targets],
                dtype=torch.long,
            )
            .reshape(-1, 2)
            .T
        )
        state_type = torch.zeros(num_nodes, dtype=torch.long)
        if self.node_source:
            state_type[self.node_source] = NodeStateType.NodeSource
        if self.edge_source:
            state_type[self.edge_source] = NodeStateType.EdgeSource
        if self.frontier:
            state_type[self.frontier] = NodeStateType.Frontier + torch.arange(
                len(self.frontier)
            )
        return gd.Data(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            node_state_type=state_type,
            non_edge_index=non_edge_index,
            num_nodes=num_nodes,
        )


def graph_bfs_trajectory(G):
    num_nodes = len(G)
    init = random.randint(0, num_nodes - 1)  # TODO: backward model
    node_mapping = [-1] * num_nodes
    node_mapping[init] = 0
    frontier = deque([init])

    state = GraphState.initial_state()
    action = GraphAction(
        GraphActionType.Init, node_type=G.nodes[init]["node_type"], target=init
    )
    trajectory = [state, action]
    state = state.apply_action(action, copy=True)
    trajectory.append(state)

    def next_state(state, action):
        state = state.apply_action(action, copy=True)
        trajectory.extend([action, state])
        return state

    while state.state_type != GraphStateType.Terminal:  # Loop: StopNode
        # u: node_source
        u = frontier.popleft()
        u_neighbors = list(G[u])
        random.shuffle(u_neighbors)
        while u_neighbors:  # Loop: AddNode
            # v: edge_source
            v = u_neighbors.pop()  # TODO: backward model

            if node_mapping[v] == -1:
                node_mapping[v] = len(state.graph)
                frontier.append(v)
                action = GraphAction(
                    GraphActionType.AddNode,
                    node_type=G.nodes[v]["node_type"],
                    edge_type=G.edges[u, v]["edge_type"],
                )
                state = next_state(state, action)

                edge_targets = list(set(frontier) - set([u, v]))
                while edge_targets:  # Loop: AddEdge
                    t = edge_targets.pop()  # TODO: backward model
                    if (v, t) in G.edges:
                        target = node_mapping[t]
                        target_index = state.edge_targets.index(target)
                        action = GraphAction(
                            GraphActionType.AddEdge,
                            target=target_index,
                            edge_type=G.edges[v, t]["edge_type"],
                        )
                        state = next_state(state, action)

                if state.state_type == GraphStateType.EdgeLevel:
                    state = next_state(state, GraphAction(GraphActionType.StopEdge))

        state = next_state(state, GraphAction(GraphActionType.StopNode))

    return dict(states=trajectory[::2], actions=trajectory[1::2])
