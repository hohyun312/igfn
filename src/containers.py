from typing import List, Optional
from dataclasses import dataclass, field
from collections import deque

import enum
import networkx as nx
import torch

from src.wl_hash import weisfeiler_lehman_graph_hash


class GraphActionType(enum.Enum):
    First = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    StopEdge = enum.auto()
    StopNode = enum.auto()


class GraphStateType(enum.Enum):
    Initial = enum.auto()
    NodeLevel = enum.auto()
    EdgeLevel = enum.auto()
    Terminal = enum.auto()


class NodeStateType(enum.Enum):
    Ancestor = enum.auto()
    NodeSource = enum.auto()
    EdgeSource = enum.auto()
    Frontier = enum.auto()


@dataclass
class GraphAction:
    action_type: GraphActionType
    target: Optional[int] = None  # AddEdge, First(B), AddNode(B)
    node_type: Optional[int] = None  # Init(F), AddNode(F, B)
    edge_type: Optional[int] = None  # AddEdge, AddNode(F, B)
    node_label: Optional[int] = None  # First, AddNode

    def __repr__(self):
        attr = ", ".join(
            [
                n + ": " + str(getattr(self, n))
                for n in ("target", "node_type", "edge_type", "node_label")
                if getattr(self, n) is not None
            ]
        )
        if attr:
            attr = ", " + attr
        return f"{self.__class__.__name__}(action_type: {self.action_type.name}{attr})"


@dataclass
class GraphState:
    state_type: GraphStateType
    graph: nx.Graph
    node_source: Optional[int] = None  # EdgeLevel, NodeLevel
    edge_source: Optional[int] = None  # EdgeLevel
    queue: deque[int] = field(default_factory=deque)

    _node_hashes: Optional[dict] = None
    _q_index: int = 0

    def is_sane(self):
        if self.state_type == GraphStateType.EdgeLevel:
            assert self.edge_source is not None
            assert self.node_source is not None
            assert self.edge_source not in self.queue
            assert self.node_source not in self.queue
        elif self.state_type == GraphStateType.NodeLevel:
            assert self.node_source is not None
        return True

    def edge_targets(self):
        return list(self.queue)[self._q_index :]

    def hash_nodes(self, iterations=4):
        self._node_hashes = weisfeiler_lehman_graph_hash(
            self.graph,
            iterations=iterations,
            edge_attr="edge_type",
            node_attr="node_type",
        )

    def terminate(self):
        self.state_type = GraphStateType.Terminal
        self.node_source = None
        self.edge_source = None
        self.queue = deque()

    def __repr__(self):
        attr = ", ".join(
            [
                n + ": " + str(getattr(self, n))
                for n in ("node_source", "edge_source")
                if getattr(self, n) is not None
            ]
        )
        if self.queue:
            attr = attr + f", queue: {self.queue}"
        if attr:
            attr = ", " + attr
        return f"{self.__class__.__name__}(state_type: {self.state_type.name}, num_nodes: {len(self.graph)}, num_edges: {len(self.graph.edges)}{attr})"


@dataclass
class Trajectory:
    states: List[GraphState] = field(default_factory=list)
    actions: List[GraphState] = field(default_factory=list)

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    @property
    def last_state(self):
        return self.states[-1]

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {len(self)})"

    def to_conditioned_states(self, iterations=4):
        cond = self.last_state
        cond.hash_nodes(iterations=iterations)
        states = self.states[:-1]
        for state in states:
            state.cond = cond

    def is_sane(self):
        states, actions = self.states, self.actions
        assert len(states) == len(actions) + 1

        for i in range(len(actions)):
            s, s_, a = states[i], states[i + 1], actions[i]

            if a.action_type == GraphActionType.First:
                assert s.state_type == GraphStateType.Initial
            elif a.action_type in [GraphActionType.AddNode, GraphActionType.StopNode]:
                assert s.state_type == GraphStateType.NodeLevel
            elif a.action_type in [GraphActionType.AddEdge, GraphActionType.StopEdge]:
                assert s.state_type == GraphStateType.EdgeLevel
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
        return f"{self.__class__.__name__}(sizes: {[len(t) for t in self.data]})"


@dataclass
class SortedStates:
    states: List[GraphState]
    initial: List[GraphState] = field(default_factory=list)
    node_level: List[GraphState] = field(default_factory=list)
    edge_level: List[GraphState] = field(default_factory=list)
    terminal: List[GraphState] = field(default_factory=list)
    initial_index: List[int] = field(default_factory=list)
    node_level_index: List[int] = field(default_factory=list)
    edge_level_index: List[int] = field(default_factory=list)
    terminal_index: List[int] = field(default_factory=list)
    sizes: List[int] = field(init=False)

    def __post_init__(self):
        for i, state in enumerate(self.states):
            if state.state_type == GraphStateType.Initial:
                self.initial.append(state)
                self.initial_index.append(i)
            elif state.state_type == GraphStateType.NodeLevel:
                self.node_level.append(state)
                self.node_level_index.append(i)
            elif state.state_type == GraphStateType.EdgeLevel:
                self.edge_level.append(state)
                self.edge_level_index.append(i)
            elif state.state_type == GraphStateType.Terminal:
                self.terminal.append(state)
                self.terminal_index.append(i)
            else:
                raise ValueError

        self.sizes = [
            len(self.initial),
            len(self.node_level),
            len(self.edge_level),
            len(self.terminal),
        ]

    def tolist(self):
        return self.initial + self.node_level + self.edge_level + self.terminal

    def indices(self):
        return torch.LongTensor(
            self.initial_index
            + self.node_level_index
            + self.edge_level_index
            + self.terminal_index
        )

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes: {self.sizes})"
