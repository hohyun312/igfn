from typing import List, Optional
import random
import networkx as nx
from copy import deepcopy
from collections import deque

import torch
import torch_geometric.data as gd

from src.containers import (
    GraphAction,
    GraphActionType,
    GraphState,
    GraphStateType,
    NodeStateType,
    Trajectory,
    SortedStates,
)
from src.wl_hash import weisfeiler_lehman_graph_hash
from src import index_utils


def _get_unique_nodes(graph: nx.Graph, iterations: int = 4):
    """
    Unique nodes according to weisfeiler lehman hash.
    Lower node label is selected when there are multiple choices.
    """
    node_labels = weisfeiler_lehman_graph_hash(
        graph,
        iterations=iterations,
        edge_attr="edge_type",
        node_attr="node_type",
    )
    visited = set()
    output = []
    for i in sorted(node_labels):
        if node_labels[i] not in visited:
            visited.add(node_labels[i])
            output.append(i)
    return output


def _hash_nodes(graph: nx.Graph, iterations: int = 4):
    """
    Map each node to an integer according to weisfeiler lehman hash.
    """
    node_labels = weisfeiler_lehman_graph_hash(
        graph,
        iterations=iterations,
        edge_attr="edge_type",
        node_attr="node_type",
    )
    hash2num = {}
    output = []
    for i in sorted(node_labels):
        label = node_labels[i]
        if label not in hash2num:
            hash2num[label] = len(hash2num)

        output.append(hash2num[label])
    return output


class GraphEnv:
    GraphAction = GraphAction
    GraphActionType = GraphActionType
    GraphState = GraphState
    GraphStateType = GraphStateType
    NodeStateType = NodeStateType
    Trajectory = Trajectory

    def __init__(self, num_node_types: int, num_edge_types: int) -> None:
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

    def initial_state(self):
        return GraphState(GraphStateType.Initial, nx.Graph())

    def step(self, state: GraphState, action: GraphAction, copy: bool = True):
        if copy:
            state = deepcopy(state)
        else:
            state = state

        if action.action_type == GraphActionType.Init:
            node_label = (
                action.node_label if action.node_label is not None else len(state.graph)
            )
            state.graph.add_node(node_label, node_type=action.node_type)
            state = GraphState(
                GraphStateType.NodeLevel, graph=state.graph, node_source=node_label
            )

        elif action.action_type == GraphActionType.AddNode:
            assert action.node_type is not None
            assert action.edge_type is not None
            node_label = (
                action.node_label if action.node_label is not None else len(state.graph)
            )
            state.graph.add_node(node_label, node_type=action.node_type)
            state.graph.add_edge(
                state.node_source, node_label, edge_type=action.edge_type
            )
            state = GraphState(
                GraphStateType.EdgeLevel,
                graph=state.graph,
                node_source=state.node_source,
                edge_source=node_label,
                frontier=state.frontier,
            )
            if len(state.frontier) == 0:
                action = GraphAction(GraphActionType.StopEdge)
                state = self.step(state, action, copy=False)

        elif action.action_type == GraphActionType.AddEdge:
            assert action.edge_type is not None
            assert action.target is not None
            targets = state.edge_targets
            state.graph.add_edge(
                state.edge_source,
                targets[action.target],
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
                state = self.step(state, action, copy=False)

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

    def graph_bfs_trajectory(self, graph: nx.Graph, new_index: bool = True):
        num_nodes = len(graph)
        init = random.randint(0, num_nodes - 1)  # TODO: backward model
        node_mapping = [-1] * num_nodes
        node_mapping[init] = 0
        frontier = deque([init])

        state = self.initial_state()

        action = GraphAction(
            GraphActionType.Init,
            node_type=graph.nodes[init]["node_type"],
            node_label=0 if new_index else init,
        )
        trajectory = [state]

        def next_state(state, action):
            state = self.step(state, action, copy=True)
            trajectory.extend([action, state])
            return state

        state = next_state(state, action)

        while state.state_type != GraphStateType.Terminal:  # Loop: StopNode
            # u: node_source
            u = frontier.popleft()
            u_neighbors = list(graph[u])
            random.shuffle(u_neighbors)
            while u_neighbors:  # Loop: AddNode
                # v: edge_source
                v = u_neighbors.pop()  # TODO: backward model

                if node_mapping[v] == -1:
                    node_mapping[v] = len(state.graph)
                    frontier.append(v)
                    action = GraphAction(
                        GraphActionType.AddNode,
                        node_type=graph.nodes[v]["node_type"],
                        edge_type=graph.edges[u, v]["edge_type"],
                        node_label=len(state.graph) if new_index else v,
                    )
                    state = next_state(state, action)

                    edge_targets = list(set(frontier) - set([u, v]))
                    while edge_targets:  # Loop: AddEdge
                        t = edge_targets.pop()  # TODO: backward model
                        if (v, t) in graph.edges:
                            target_node = node_mapping[t] if new_index else t
                            target_index = state.edge_targets.index(target_node)

                            action = GraphAction(
                                GraphActionType.AddEdge,
                                target=target_index,
                                edge_type=graph.edges[v, t]["edge_type"],
                            )
                            state = next_state(state, action)

                    if state.state_type == GraphStateType.EdgeLevel:
                        state = next_state(state, GraphAction(GraphActionType.StopEdge))

            state = next_state(state, GraphAction(GraphActionType.StopNode))

        return Trajectory(states=trajectory[::2], actions=trajectory[1::2])

    def follow_actions(self, actions: List[GraphAction]):
        states = [self.initial_state()]
        for act in actions:
            states += [self.step(states[-1], act)]
        return Trajectory(states, actions)

    def to_tensor_graph(self, state: GraphState):
        num_nodes = len(state.graph.nodes())
        node_type = torch.tensor(
            [attr["node_type"] for _, attr in sorted(state.graph.nodes(data=True))],
            dtype=torch.long,
        )
        node_label = torch.tensor(
            sorted(state.graph.nodes),
            dtype=torch.long,
        )
        edge_index = (
            torch.tensor(
                [e for i, j in state.graph.edges for e in [(i, j), (j, i)]],
                dtype=torch.long,
            )
            .reshape(-1, 2)
            .T
        )
        edge_type = torch.tensor(
            [
                (attr["edge_type"], attr["edge_type"])
                for _, _, attr in state.graph.edges(data=True)
            ],
            dtype=torch.long,
        ).flatten()
        non_edge_index = (
            torch.tensor(
                [(state.edge_source, v) for v in state.edge_targets],
                dtype=torch.long,
            )
            .reshape(-1, 2)
            .T
        )

        # Mapping node indices. This is to ensure label invariance.
        mapping = index_utils.relabel_mapping(edge_index)
        node_mapping = index_utils.inv_relabel_mapping(mapping)
        edge_index = mapping[edge_index]
        non_edge_index = mapping[non_edge_index]

        node_state_type = torch.zeros(num_nodes, dtype=torch.long)
        frontier_order = torch.full((num_nodes,), -1, dtype=torch.long)
        if state.node_source is not None:
            node_source = mapping[state.node_source] if len(mapping) > 1 else 0
            node_state_type[node_source] = NodeStateType.NodeSource.value - 1
        if state.edge_source is not None:
            edge_source = mapping[state.edge_source]
            node_state_type[edge_source] = NodeStateType.EdgeSource.value - 1
        if state.frontier:
            frontier = mapping[state.frontier]
            node_state_type[frontier] = NodeStateType.Frontier.value - 1
            frontier_order[frontier] = torch.arange(len(frontier))

        return gd.Data(
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            node_state_type=node_state_type,
            non_edge_index=non_edge_index,
            num_non_edges=non_edge_index.shape[1],
            state_type=state.state_type.value,
            frontier_order=frontier_order,
            node_mapping=node_mapping,
            node_label=node_label,
        )

    def collate(
        self, states: List[GraphState], cond_states: Optional[List[GraphState]] = None
    ):
        if cond_states is None:
            data_list = [self.to_tensor_graph(s) for s in states]
        else:
            data_list = [None] * len(states)
            for i in range(len(states)):
                s, c = states[i], cond_states[i]
                if s.state_type == GraphStateType.Initial:
                    g = self.to_tensor_graph(c)
                    # pretend to be inital state
                    g.state_type = GraphStateType.Initial.value
                    # g.wl_node_index = torch.LongTensor(_get_unique_nodes(c.graph))
                else:
                    g = self.to_tensor_graph(s)
                    # g.wl_node_index = torch.LongTensor([])
                data_list[i] = g

        graphs = gd.Batch.from_data_list(data_list)
        graphs.is_init = graphs.state_type == GraphStateType.Initial.value
        graphs.is_nodelv = graphs.state_type == GraphStateType.NodeLevel.value
        graphs.is_edgelv = graphs.state_type == GraphStateType.EdgeLevel.value
        graphs.is_terminal = graphs.state_type == GraphStateType.Terminal.value

        graphs.init_index = graphs.is_init.nonzero().flatten()
        graphs.nodelv_index = graphs.is_nodelv.nonzero().flatten()
        graphs.edgelv_index = graphs.is_edgelv.nonzero().flatten()
        graphs.terminal_index = graphs.is_terminal.nonzero().flatten()
        return graphs

    def collate(self, sorted_states: SortedStates):
        data_list = [self.to_tensor_graph(s) for s in sorted_states.tolist()]
        graphs = gd.Batch.from_data_list(data_list)
        graphs.sptr = torch.cumsum(torch.LongTensor([0] + sorted_states.sizes), dim=0)
        return graphs
