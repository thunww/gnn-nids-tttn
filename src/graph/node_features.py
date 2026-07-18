import networkx as nx
import numpy as np


def compute_node_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Tinh dac trung cho tung dinh dua tren vi tri cua no trong do thi:
    bac vao, bac ra, PageRank, he so phan cum.

    Tra ve mang shape (num_nodes, 4).
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph.to_undirected())

    features = np.zeros((num_nodes, 4), dtype="float32")
    for i in range(num_nodes):
        features[i, 0] = in_degree.get(i, 0)
        features[i, 1] = out_degree.get(i, 0)
        features[i, 2] = pagerank.get(i, 0.0)
        features[i, 3] = clustering.get(i, 0.0)

    return features
