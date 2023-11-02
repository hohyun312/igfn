from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import enum

import torch
import torch_geometric.data as gd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

from rdkit.Chem.rdchem import BondType
from rdkit import Chem
from src.wl_hash import weisfeiler_lehman_graph_hash


CMAP = plt.get_cmap("tab20")


class ActionType(enum.Enum):
    First = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    StopEdge = enum.auto()
    StopNode = enum.auto()


class StateType(enum.Enum):
    Initial = enum.auto()
    NodeLevel = enum.auto()
    EdgeLevel = enum.auto()
    Terminal = enum.auto()


@dataclass
class Action:
    action_type: ActionType
    target: Optional[int] = None  # AddEdge, First(B), AddNode(B)
    node_type: Optional[int] = None  # Init(F), AddNode(F, B)
    edge_type: Optional[int] = None  # AddEdge, AddNode(F, B)

    def __repr__(self):
        attr = ", ".join(
            [
                n + ": " + str(getattr(self, n))
                for n in ("target", "node_type", "edge_type")
                if getattr(self, n) is not None
            ]
        )
        if attr:
            attr = ", " + attr
        return f"{self.__class__.__name__}(action_type: {self.action_type.name}{attr})"


@dataclass
class State:
    """
    State class for BFS-ordered generation.
    """

    state_type: StateType

    # graph related
    node_type: list[int] = field(default_factory=list)
    edge_type: list[int] = field(default_factory=list)
    edge_list: list[tuple[int, int]] = field(default_factory=list)

    # BFS related: environment controls these variables
    node_order: Optional[int] = field(default_factory=list)
    node_source: Optional[int] = None  # EdgeLevel, NodeLevel
    edge_source: Optional[int] = None  # EdgeLevel
    target_range: Optional[tuple[int, int]] = None

    adj: list[list[int]] = field(default_factory=list)
    e2t: dict[tuple, int] = field(default_factory=dict)

    def __post_init__(self):
        self.build_adjacency_list()
        self.build_edge_to_type()
        
    @property
    def num_nodes(self):
        return len(self.node_type)

    @property
    def num_edges(self):
        return len(self.edge_type)

    @property
    def degree(self):
        return [len(n) for n in self.adj]

    def add_node(self, node_type: int):
        assert isinstance(node_type, int)
        self.node_order.append(self.num_nodes)
        self.node_type.append(node_type)
        self.adj.append([])

    def add_edge(self, u: int, v: int, edge_type: int):
        assert isinstance(edge_type, int)
        self.edge_list.extend([(u, v), (v, u)])  # undirected
        self.edge_type.extend([edge_type, edge_type])
        self.adj[v].append(u)
        self.adj[u].append(v)
        self.e2t[(u, v)] = edge_type
        self.e2t[(v, u)] = edge_type

    def queue(self):
        """
        Nodes that are generated are queued to be processed as a node_source.
        These are nodes between node_source and edge_source (if exist).
        """
        j = self.num_nodes if self.edge_source is None else self.edge_source
        i = j if self.node_source is None else self.node_source + 1
        return list(range(i, j))

    @property
    def edge_target(self):
        """
        Edge targets are nodes that can be connected with edge_source at EdgeLevel phase generation.
        """
        return [] if self.target_range is None else list(range(*self.target_range))

    def build_adjacency_list(self):
        neighbor = [[] for _ in range(self.num_nodes)]
        for u, v in self.edge_list:
            neighbor[u].append(v)
        self.adj = neighbor
        return self

    def build_edge_to_type(self):
        self.e2t = dict(zip(self.edge_list, self.edge_type))
        return self

    def get_neighboring_edge_attr(self):
        attr = [[] for _ in range(self.num_nodes)]
        for (u, v), t in zip(self.edge_list, self.edge_type):
            attr[u].append(t)
        return attr

    def get_clustering_coef(self):
        cluster_coef = []
        for nbrs in self.adj:
            if len(nbrs) <= 1:
                coef = 0.0
            else:
                t = 0.0
                u_nbrs = set(nbrs)
                seen = set()
                for v in u_nbrs:
                    seen.add(v)
                    v_nbrs = set(self.adj[v]) - seen
                    t += len(u_nbrs & v_nbrs)
                d = len(u_nbrs)
                coef = 2 * t / (d * (d - 1))
            cluster_coef.append(coef)
        return cluster_coef

    def hval(self, iterations=4):
        """hash value: integer values where structually similar nodes are mapped to the same value."""
        degree = self.degree
        node_attrs = [degree, self.node_type, self.get_clustering_coef()]
        node_labels = weisfeiler_lehman_graph_hash(
            self.adj,
            node_attrs,
            self.get_neighboring_edge_attr(),
            self.num_nodes,
            iterations=iterations,
        )
        items = sorted(node_labels.items(), reverse=True)
        integer_hash = {v: k for k, v in items}

        node_hash = [None] * self.num_nodes
        for node, hash_value in node_labels.items():
            node_hash[node] = integer_hash[hash_value]

        return node_hash

    def count_automorphisms(self):
        if self.edge_list:
            g = ig.Graph(edges=self.edge_list)
            return g.count_automorphisms_vf2(color=self.node_type)
        else:
            return 1

    def to_tensor(self):
        node_type = torch.LongTensor(self.node_type)
        edge_type = torch.LongTensor(self.edge_type)
        edge_index = torch.LongTensor(self.edge_list).reshape(-1, 2).t().contiguous()
        non_edge_index = (
            torch.LongTensor(
                [(self.edge_source, v) for v in self.edge_target],
            )
            .reshape(-1, 2)
            .t()
            .contiguous()
        )
        node_order = torch.LongTensor(self.node_order)

        graph_state_id = self.state_type.value - 1
        node_state_id = torch.zeros(self.num_nodes, dtype=torch.long)
        if self.node_source is not None:
            node_state_id[self.node_source] = 1
            node_state_id[self.node_source + 1 :].fill_(2)
        if self.edge_source is not None:
            node_state_id[self.edge_source] = 3

        return gd.Data(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            non_edge_index=non_edge_index,
            num_non_edges=non_edge_index.size(1),
            node_order=node_order,
            graph_state_id=graph_state_id,
            node_state_id=node_state_id,
        )

    def to_nx(self):
        graph = nx.Graph()

        for i in range(self.num_nodes):
            graph.add_node(i, node_type=self.node_type[i])

        edge_list = self.edge_list[::2]
        edge_type = self.edge_type[::2]
        for i, (u, v) in enumerate(edge_list):
            graph.add_edge(u, v, edge_type=edge_type[i])

        return graph

    @classmethod
    def from_nx(cls, graph):
        edges = sorted(graph.edges(data=True))
        nodes = sorted(graph.nodes(data=True))
        node_type = [attr.get("node_type", 0) for i, attr in nodes]
        edge_type = sum([[attr.get("edge_type", 0)] * 2 for u, v, attr in edges], [])
        edge_list = [e for u, v, _ in edges for e in [(u, v), (v, u)]]
        return cls(
            StateType.Terminal, node_type, edge_type, edge_list
        )

    def visualize(self, figsize=(3, 3), node_color="node_type", bfs_root=None, **kargs):
        def bfs_order(neighbors, root=0):
            i = root
            queue = deque([i])
            depth = [-1] * len(neighbors)
            depth[i] = 0
            node_order = [i]
            while queue:
                i = queue.popleft()
                for j in neighbors[i]:
                    if depth[j] == -1:
                        depth[j] = depth[i] + 1
                        queue.append(j)
                        node_order.append(j)
            return depth, node_order

        def bfs_pos(depth, node_order):
            d_counts = Counter(depth)
            cur = d_counts.copy()
            pos = {}
            for i in node_order:
                d = depth[i]
                x = (cur[d] + 0.5) / d_counts[d]
                pos[i] = (-1 + x, -d)
                cur[d] += 1
            return pos

        graph = self.to_nx()
        if node_color == "node_type":
            node_color = [CMAP(x) for x in self.node_type]
        else:
            node_color = [CMAP(x) for x in self.hval]
        width = [e["edge_type"] ** 2 + 1 for _, _, e in graph.edges(data=True)]

        pos = None
        if bfs_root is not None:
            depth, node_order = bfs_order(self.adj, bfs_root)
            pos = bfs_pos(depth, node_order)

        if figsize is not None:
            plt.figure(figsize=figsize)
        return nx.draw(graph, pos=pos, node_color=node_color, width=width, **kargs)

    def __repr__(self):
        attr = ", ".join(
            [
                n + ": " + str(getattr(self, n))
                for n in ("node_source", "edge_source")
                if getattr(self, n) is not None
            ]
        )
        if attr:
            attr = ", " + attr
        return f"{self.__class__.__name__}(state_type: {self.state_type.name}, num_nodes: {self.num_nodes}, num_edges: {self.num_edges}{attr})"


class BatchedState:
    def __init__(self, states=None):
        self.data = []
        self.sort_idx = defaultdict(list)
        self.size_by_type = [0, 0, 0, 0]
        self.total_size = 0

        if states is not None:
            for state in states:
                self.add(state)

    def add(self, state):
        if state.state_type == StateType.Initial:
            key = "initial"
        elif state.state_type == StateType.NodeLevel:
            key = "node_level"
        elif state.state_type == StateType.EdgeLevel:
            key = "edge_level"
        elif state.state_type == StateType.Terminal:
            key = "terminal"
        else:
            raise ValueError

        self.data.append(state)
        self.sort_idx[key].append(self.total_size)
        self.size_by_type[state.state_type.value - 1] += 1
        self.total_size += 1

    def num_nodes(self):
        return torch.LongTensor([x.num_nodes for x in self.data])

    def adjacency_lists(self):
        return [s.adj for s in self.data]

    def count_automorphisms(self):
        return torch.FloatTensor([s.count_automorphisms() for s in self.data])

    def get_states(self):
        return self.data

    def get_node_order(self):
        return torch.LongTensor(sum([x.node_order for x in self.data], []))

    def sort_states(self):
        return (
            [self.data[i] for i in self.sort_idx["initial"]]
            + [self.data[i] for i in self.sort_idx["node_level"]]
            + [self.data[i] for i in self.sort_idx["edge_level"]]
            + [self.data[i] for i in self.sort_idx["terminal"]]
        )

    def sort_indices(self):
        return torch.LongTensor(
            self.sort_idx["initial"]
            + self.sort_idx["node_level"]
            + self.sort_idx["edge_level"]
            + self.sort_idx["terminal"]
        )

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.total_size

    def __repr__(self):
        return f"{self.__class__.__name__}(size_by_type: {self.size_by_type})"


class Trajectory:
    def __init__(self, states=None, actions=None):
        self._states: list[State] = []
        self._actions: list[State] = []

        if states is not None:
            self._states = states
        if actions is not None:
            self._actions = actions

    def add_state(self, state):
        self._states.append(state)

    def add_action(self, action):
        self._actions.append(action)

    def add_state_and_action(self, state, action):
        self._states.append(state)
        self._actions.append(action)

    def get_states(self):
        return BatchedState(self._states[:-1])

    def get_actions(self):
        return self._actions

    def get_last_state(self):
        return self._states[-1]

    def __len__(self):
        return len(self._actions)

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {len(self)})"


class BatchedTrajectory:
    def __init__(self, traj: list[Trajectory] = None):
        self.data = []

        if traj is not None:
            self.data = traj

    def add(self, traj):
        self.data.append(traj)

    def get_states(self):
        return BatchedState(sum([t._states[:-1] for t in self.data], []))

    def get_actions(self):
        return sum([t._actions for t in self.data], [])

    def get_last_state(self):
        return BatchedState([t.get_last_state() for t in self.data])

    def get_length(self):
        return torch.LongTensor([len(t) for t in self.data])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}(sizes: {[len(t) for t in self.data]})"