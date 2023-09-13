import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def connect_graphs(G):
    components = [c for c in nx.connected_components(G)]
    for _ in range(len(components) - 1):
        i, j = np.random.choice(range(len(components)), size=2, replace=False)
        c1, c2 = components[i], components[j]
        node1 = np.random.choice(list(c1))
        node2 = np.random.choice(list(c2))
        G.add_edge(node1, node2)

        components[i] = c1.union(c2)
        components.pop(j)
    return G


def gnp_connected(n, p):
    G = nx.gnp_random_graph(n, p)
    return connect_graphs(G)


def n_community(c_sizes, p=0.3, p_inter=0.01):
    # Adapted from https://github.com/JiaxuanYou/graph-generation
    graphs = [gnp_connected(c_sizes[i], p) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)

    for i in range(len(communities)):
        subG = communities[i]
        for _, attr in subG.nodes(data=True):
            attr["node_type"] = i

    G = connect_graphs(G)

    for _, _, attr in G.edges(data=True):
        attr["edge_type"] = np.random.randint(0, 2)

    return G


rng = np.random.default_rng(0)
color_map = rng.uniform(0, 1, size=(100, 3))


def draw_graph(G, figsize=(3, 3), with_labels=False):
    node_color = None
    if "node_type" in G.nodes[list(G.nodes)[0]]:
        labels = [attr["node_type"] for _, attr in G.nodes(data=True)]
        node_color = [tuple(rgb) for rgb in color_map[labels]]

    edge_data = list(G.edges(data=True))
    edge_color = None
    if len(edge_data) > 0 and "edge_type" in edge_data[0][2]:
        labels = [50 + attr["edge_type"] for *_, attr in edge_data]
        edge_color = [tuple(rgb) for rgb in color_map[labels]]

    plt.figure(figsize=figsize)
    return nx.draw(
        G,
        node_color=node_color,
        edge_color=edge_color,
        with_labels=with_labels,
    )
