import networkx as nx
import numpy as np


def compute_structural_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Tinh dac trung cau truc cho tung dinh dua tren vi tri cua no trong do thi:
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


def aggregate_edge_features_per_node(edge_index: np.ndarray, edge_attr: np.ndarray, num_nodes: int) -> np.ndarray:
    """Tong hop (trung binh) dac trung cua cac canh ke voi tung node (ca chieu vao lan ra).

    Muc dich: node hien chi co dac trung cau truc thuan tuy (bac, PageRank...), khong biet
    gi ve NOI DUNG luong mang that (byte, thoi luong...) di qua no. Nghien cuu (N2V-EGS-PCA,
    arXiv:2404.10800) xac nhan "bo dac trung day du gom ca canh lan node" la yeu to quan
    trong nhat de dat F1-macro cao. Node2Vec/line-graph (cach lam giau node "chuan" hon trong
    tai lieu) khong kha thi trong pipeline theo-cua-so hien tai (xem docs/decisions.md) --
    day la cach thay the thuc te: tinh trung binh truc tiep, khong can huan luyen gi them.

    Tra ve mang shape (num_nodes, so_dac_trung_canh).
    """
    num_edge_features = edge_attr.shape[1]
    sums = np.zeros((num_nodes, num_edge_features), dtype="float64")
    counts = np.zeros(num_nodes, dtype="float64")

    src, dst = edge_index[0], edge_index[1]
    np.add.at(sums, src, edge_attr)
    np.add.at(sums, dst, edge_attr)
    np.add.at(counts, src, 1)
    np.add.at(counts, dst, 1)

    counts = np.clip(counts, 1, None)[:, None]
    means = sums / counts

    return means.astype("float32")


def compute_node_features(edge_index: np.ndarray, edge_attr: np.ndarray, num_nodes: int) -> np.ndarray:
    """Ghep dac trung cau truc (4 chieu) + dac trung tong hop tu canh ke (so_dac_trung_canh
    chieu) thanh dac trung node day du.

    Tra ve mang shape (num_nodes, 4 + so_dac_trung_canh).
    """
    structural = compute_structural_features(edge_index, num_nodes)
    edge_agg = aggregate_edge_features_per_node(edge_index, edge_attr, num_nodes)
    return np.concatenate([structural, edge_agg], axis=1)
