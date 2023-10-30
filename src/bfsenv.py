from typing import Optional
from functools import partial
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from copy import deepcopy
import enum

import numpy as np
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

    def terminate(self):
        self.state_type = StateType.Terminal
        self.node_source = None
        self.edge_source = None
        self.target_range = None
        return self

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
        return cls(StateType.Terminal, node_type, edge_type, edge_list).build_adjacency_list()

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

    def set_node_order(self, node_order):
        split_size = torch.Size(self.num_nodes())
        node_order_split = node_order.detach().cpu().split(split_size)

        for i in range(self.total_size):
            state = self.data[i]
            state.node_order = node_order_split[i].tolist()

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
    def __init__(self):
        self.data = []

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


class MolGraph(State):
    id2atom = ["C", "N", "O", "S", "P", "F", "I", "Cl", "Br"]
    id2bond = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]  # BondType.AROMATIC

    bond2id = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    atom2id = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5, "I": 6, "Cl": 7, "Br": 8}

    num_edge_types = len(bond2id)
    num_node_types = len(atom2id)

    @classmethod
    def from_molecule(cls, molecule):
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        node_type = cls.get_node_type(molecule)
        edge_list, edge_type = cls.get_edge_list_and_type(molecule)
        return cls(StateType.Terminal, node_type, edge_type, edge_list)

    def to_molecule(self):
        mol = Chem.RWMol()
        for t in self.node_type:
            atom_symbol = self.id2atom[t]
            mol.AddAtom(Chem.Atom(atom_symbol))

        # remove duplicate edges
        edge_list, edge_type = self.edge_list[::2], self.edge_type[::2]
        for (i, j), t in zip(edge_list, edge_type):
            mol.AddBond(i, j, order=self.id2bond[t])
        return mol

    @classmethod
    def from_smiles(cls, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        return cls.from_molecule(molecule)

    @classmethod
    def get_node_type(cls, molecule):
        node_type = []
        for i in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(i)
            node_type.append(cls.atom2id[atom.GetSymbol()])
        return node_type

    @classmethod
    def get_edge_list_and_type(cls, molecule):
        edge_list = []
        edge_type = []
        for i in range(molecule.GetNumBonds()):
            bond = molecule.GetBondWithIdx(i)
            t = cls.bond2id[bond.GetBondType()]
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [(u, v), (v, u)]
            edge_type += [t, t]
        return edge_list, edge_type


class GraphBuildingEnv:
    ActionType = ActionType
    StateType = StateType
    Action = Action
    State = State
    BatchedState = BatchedState
    Trajectory = Trajectory
    BatchedTrajectory = BatchedTrajectory

    def __init__(
        self, num_node_types: int, num_edge_types: int, max_degree: int = 4
    ) -> None:
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_degree = max_degree

    def initial_state(self):
        return State(StateType.Initial)

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

    # def get_trajectory(self, input_state: State):
    #     assert input_state.node_order, "node_order must be given"
    #     root = input_state.node_order[0]
    #     rank = np.argsort(input_state.node_order)  # order_index

    #     e2t = dict(zip(input_state.edge_list, input_state.edge_type))

    #     node_mapping = [-1] * input_state.num_nodes
    #     node_mapping[root] = 0
    #     queue = deque([root])

    #     state = self.initial_state()
    #     action = Action(ActionType.First, node_type=input_state.node_type[root])

    #     traj = Trajectory()
    #     traj.add_state(state)

    #     state = self.step(state, action)
    #     traj.add_state_and_action(state, action)

    #     while state.state_type != StateType.Terminal:  # Loop until terminal
    #         # u: node_source
    #         u = queue.popleft()
    #         u_neighbors = input_state.adj[u]
    #         u_neighbors = sorted(u_neighbors, key=lambda x: rank[x])

    #         # v: edge_source
    #         for v in u_neighbors:  # Loop until StopNode
    #             if node_mapping[v] == -1:
    #                 node_mapping[v] = state.num_nodes
    #                 queue.append(v)

    #                 action = Action(
    #                     ActionType.AddNode,
    #                     node_type=input_state.node_type[v],
    #                     edge_type=e2t[(u, v)],
    #                 )
    #                 state = self.step(state, action, copy=True)
    #                 traj.add_state_and_action(state, action)

    #                 targets = [q for q in queue if q != u and q != v]
    #                 for t in targets:  # Loop: AddEdge
    #                     if state.state_type != StateType.EdgeLevel:
    #                         break
    #                     if (t, v) in e2t:
    #                         target_node = node_mapping[t]
    #                         action = Action(
    #                             ActionType.AddEdge,
    #                             target=state.edge_target.index(target_node),
    #                             edge_type=e2t[(t, v)],
    #                         )
    #                         state = self.step(state, action, copy=True)
    #                         traj.add_state_and_action(state, action)

    #                 if state.state_type == StateType.EdgeLevel:
    #                     action = Action(ActionType.StopEdge)
    #                     state = self.step(state, action, copy=True)
    #                     traj.add_state_and_action(state, action)

    #         action = Action(ActionType.StopNode)
    #         state = self.step(state, action, copy=True)
    #         traj.add_state_and_action(state, action)

    #     return traj

    def get_trajectory(self, input_state: State):
        assert input_state.node_order, "node_order must be given"
        assert input_state.state_type == StateType.Terminal

        rank = np.argsort(input_state.node_order)  # order_index
        e2t = dict(zip(input_state.edge_list, input_state.edge_type))
        input_adj = [
            sorted(input_state.adj[u], key=lambda x: rank[x])
            for u in range(input_state.num_nodes)
        ]

        root = input_state.node_order[0]
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
                        edge_type=e2t[(u, v)],
                    )
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)
                else:
                    action = Action(ActionType.StopNode)
                    state = self.step(state, action, copy=True)
                    traj.add_state_and_action(state, action)

            elif state.state_type == StateType.EdgeLevel:
                flg = False
                for i in range(*state.target_range):
                    t = out2in[i]
                    if (t, v) in e2t:
                        flg = True
                        break

                if flg:
                    target_node = in2out[t]
                    target = target_node - state.target_range[0]
                    action = Action(
                        ActionType.AddEdge,
                        target=target,
                        edge_type=e2t[(t, v)],
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

    def state_to_trajectory(self, states: BatchedState):
        traj = BatchedTrajectory()
        for x in states.data:
            traj.add(self.get_trajectory(x))
        return traj


class ForwardActionDistribution:
    def __init__(
        self,
        init_cat,
        nodelv_cat,
        edgelv_cat,
        num_edge_types,
        from_index,
    ):
        self.init_cat = init_cat
        self.nodelv_cat = nodelv_cat
        self.edgelv_cat = edgelv_cat
        self.num_edge_types = num_edge_types

        self._to_index = torch.argsort(from_index)

        self.sizes = (self.init_cat.size, nodelv_cat.size, self.edgelv_cat.size)

    def _classify_actions_by_types(self, actions):
        init_actions = [a for a in actions if a.action_type == ActionType.First]
        nodelv_actions = [
            a
            for a in actions
            if a.action_type in [ActionType.AddNode, ActionType.StopNode]
        ]
        edgelv_actions = [
            a
            for a in actions
            if a.action_type in [ActionType.AddEdge, ActionType.StopEdge]
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
        return torch.LongTensor([a.node_type for a in actions])

    def init_tensor_to_actions(self, tensor):
        return [Action(ActionType.First, node_type=i.item()) for i in tensor]

    def nodelv_actions_to_tensor(self, actions):
        return torch.LongTensor(
            [
                0
                if a.action_type == ActionType.StopNode
                else (1 + self.num_edge_types * a.node_type + a.edge_type)
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
        return torch.LongTensor(
            [
                0
                if a.action_type == ActionType.StopEdge
                else (1 + self.num_edge_types * a.target + a.edge_type)
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


if __name__ == "__main__":
    env = GraphBuildingEnv(3, 3)

    state = env.initial_state()
    actions = [
        Action(ActionType.First, node_type=0, target=0),
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

    for action in actions:
        env.step(state, action, copy=False)

    state.to_tensor()