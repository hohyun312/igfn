from typing import List
from dataclasses import dataclass, field
from collections import deque

import enum
import networkx as nx
import torch


class GraphActionType(enum.Enum):
    Init = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    StopEdge = enum.auto()
    StopNode = enum.auto()


class GraphStateType(enum.Enum):
    Initial = enum.auto()
    EdgeLevel = enum.auto()
    NodeLevel = enum.auto()
    Terminal = enum.auto()


class NodeStateType(enum.Enum):
    Ancester = enum.auto()
    NodeSource = enum.auto()
    EdgeSource = enum.auto()
    Frontier = enum.auto()


@dataclass
class GraphAction:
    action_type: GraphActionType
    target: int = None  # AddEdge, Init(B), AddNode(B)
    node_type: int = None  # Init(F), AddNode(F, B)
    edge_type: int = None  # AddEdge, AddNode(F, B)
    node_label: int = None  # Init, AddNode

    def __repr__(self):
        attr = ", ".join(
            [
                n + "=" + str(getattr(self, n))
                for n in ("target", "node_type", "edge_type")
                if getattr(self, n) is not None
            ]
        )
        return f"{self.__class__.__name__}(action_type={self.action_type.name}, {attr})"


@dataclass
class GraphState:
    state_type: GraphStateType
    graph: nx.Graph
    node_source: int = None  # EdgeLevel, NodeLevel
    edge_source: int = None  # EdgeLevel
    frontier: deque[int] = field(default_factory=deque)

    def __repr__(self):
        return f"{self.__class__.__name__}(state_type={self.state_type.name}, num_nodes={len(self.graph)}, num_edges={len(self.graph.edges)})"

    def is_sane(self):
        if self.state_type == GraphStateType.EdgeLevel:
            assert self.edge_source is not None
            assert self.node_source is not None
            assert self.edge_source not in self.frontier
            assert self.node_source not in self.frontier
        elif self.state_type == GraphStateType.NodeLevel:
            assert self.node_source is not None
        return True

    def terminate(self):
        self.state_type = GraphStateType.Terminal
        self.node_source = None
        self.edge_source = None
        self.frontier = deque()

    @property
    def edge_targets(self):
        """
        Returns a list of nodes that is not connected to `edge_source`, i.e., edge-level targets.
        """
        if self.state_type != GraphStateType.EdgeLevel:
            return []
        connected = set(self.graph[self.edge_source])
        return [i for i in self.frontier if i not in connected]


@dataclass
class Trajectory:
    states: List[GraphState] = field(default_factory=list)
    actions: List[GraphState] = field(default_factory=list)

    def add(self, state, action):
        self.states.append(state)
        self.action.append(action)

    @property
    def last_state(self):
        return self.states[-1]

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def is_sane(self):
        states, actions = self.states, self.actions
        assert len(states) == len(actions) + 1

        for i in range(len(actions)):
            s, s_, a = states[i], states[i + 1], actions[i]

            if a.action_type == GraphActionType.Init:
                assert s.state_type == GraphStateType.Initial
            elif a.action_type in [GraphActionType.AddNode, GraphActionType.StopNode]:
                assert s.state_type == GraphStateType.NodeLevel
            elif a.action_type in [GraphActionType.AddEdge, GraphActionType.StopEdge]:
                assert s.state_type == GraphStateType.EdgeLevel

            # g = env.step(s, a).graph
            # g_ = s_.graph
            # assert nx.is_isomorphic(g, g_)

        return True


class Trajectories:
    def __init__(self):
        self.data = []

    def add(self, traj):
        self.data.append(traj)

    def flatten(self):
        states = []
        actions = []
        cond_states = []
        lens = []
        for t in self.data:
            states.extend(t.states[:-1])
            actions.extend(t.actions)
            cond_states.extend([t.last_state] * (len(t) - 1))
            lens.append(len(t) - 1)
        lens = torch.LongTensor(lens)
        return states, actions, cond_states, lens

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes={[len(t) for t in self.data]})"
