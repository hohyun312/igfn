from hashlib import blake2b

def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def _init_node_labels(node_attrs, num_nodes):
    node_labels = {u: "" for u in range(num_nodes)}  
    for node_attr in node_attrs:
        node_labels = {u: node_labels[u] + "," + str(att) for u, att in enumerate(node_attr)}
    return node_labels

def _neighborhood_aggregate(node, neighbors, edge_attr, node_labels):
    """
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    label_list = []
    for nbr, att in zip(neighbors[node], edge_attr[node]):
        prefix = str(att)
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + ";".join(sorted(label_list))

def weisfeiler_lehman_step(node_labels, neighbors, edge_attr, digest_size=16):
    """
    Apply neighborhood aggregation to each node
    in the graph.
    Computes a dictionary with labels for each node.
    """
    new_labels = {}
    for node in range(len(neighbors)):
        label = _neighborhood_aggregate(node, neighbors, edge_attr, node_labels)
        new_labels[node] = _hash_label(label, digest_size)
    return new_labels

def weisfeiler_lehman_graph_hash(neighbors, node_attrs, edge_attr, num_nodes, iterations=3, digest_size=16):
    node_labels = _init_node_labels(node_attrs, num_nodes)

    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(node_labels, neighbors, edge_attr, digest_size)
    return node_labels