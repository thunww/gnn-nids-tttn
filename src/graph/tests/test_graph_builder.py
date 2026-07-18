import pandas as pd

from graph.edges import build_edges
from graph.node_features import compute_node_features
from graph.nodes import build_node_ids


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "L4_SRC_PORT": [1111, 1111, 2222],
            "IPV4_DST_ADDR": ["10.0.0.9", "10.0.0.8", "10.0.0.9"],
            "L4_DST_PORT": [80, 80, 443],
            "IN_BYTES": [100.0, 200.0, 50.0],
        }
    )


def test_build_node_ids_dedups_repeated_endpoints():
    df = _sample_df()
    node_id, _, _ = build_node_ids(df, "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")

    # 5 dinh duy nhat: 10.0.0.1:1111, 10.0.0.2:2222, 10.0.0.9:80, 10.0.0.8:80, 10.0.0.9:443
    assert len(node_id) == 5


def test_build_edges_shape_matches_rows():
    df = _sample_df()
    node_id, src_key, dst_key = build_node_ids(df, "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")
    edge_index, edge_attr = build_edges(df, node_id, src_key, dst_key, ["IN_BYTES"])

    assert edge_index.shape == (2, len(df))
    assert edge_attr.shape == (len(df), 1)
    # dinh dau tien cua canh dau tien phai la id cua "10.0.0.1:1111"
    assert edge_index[0, 0] == node_id["10.0.0.1:1111"]
    assert edge_index[1, 0] == node_id["10.0.0.9:80"]


def test_compute_node_features_shape_and_degree():
    df = _sample_df()
    node_id, src_key, dst_key = build_node_ids(df, "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")
    edge_index, _ = build_edges(df, node_id, src_key, dst_key, ["IN_BYTES"])
    features = compute_node_features(edge_index, len(node_id))

    assert features.shape == (len(node_id), 4)
    # "10.0.0.1:1111" la nguon cua 2 canh -> out_degree = 2
    out_degree_col = 1
    assert features[node_id["10.0.0.1:1111"], out_degree_col] == 2
