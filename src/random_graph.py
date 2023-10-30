
import numpy as np
import networkx as nx
from src.bfsenv import BatchedState, State

def grid_graph(size=1):
    graphs = []
    for _ in range(size):
        i, j = np.random.randint(2, 8, size=2)
        G = nx.grid_2d_graph(i, j)
        G = nx.relabel_nodes(G, dict(zip(G.nodes, range(len(G)))))
        graphs.append(G)
    return BatchedState([State.from_nx(G) for G in graphs])

def binomial_tree_graph(size=1):
    graphs = []
    for _ in range(size):
        i = np.random.randint(1, 6)
        graphs.append(nx.binomial_tree(i))
    return BatchedState([State.from_nx(G) for G in graphs])

def regular_graph(size=1):
    graphs = []
    for _ in range(size):
        n = np.random.randint(6, 12)
        d = np.random.randint(2, 4)

        if (n * d) % 2 == 1:
            n -= 1

        G = nx.random_regular_graph(d, n)

        if not nx.is_connected(G):
            nodes = next(nx.connected_components(G))
            G = nx.subgraph(G, nodes)
            G = nx.relabel_nodes(G, dict(zip(G.nodes, range(len(G)))))
    
        graphs.append(G)
    output = BatchedState([State.from_nx(G) for G in graphs])
    return output

def perturb(batch, p=0.2):
    '''
    Randomly remove one edge with probability p.
    '''
    perturbed = []
    for state in batch:
        if np.random.random() < p:
            edge = np.random.choice(state.num_edges)
            u, v = state.edge_list[edge]
            degree = [len(nbr) for nbr in state.adj]
            if (degree[u] >= 2) and (degree[v] >= 2):
                state = state.copy()
                edge = edge - edge % 2
                i, j = edge, edge + 1
                state.edge_type = state.edge_type[:i] + state.edge_type[j+1:]
                state.edge_list = state.edge_list[:i] + state.edge_list[j+1:]
        perturbed.append(state)
    return BatchedState(perturbed)
