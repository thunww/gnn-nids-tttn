import numpy as np
import pandas as pd


def build_edges(
    df: pd.DataFrame,
    node_id: dict[str, int],
    src_key: pd.Series,
    dst_key: pd.Series,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Dung danh sach canh tu tung dong (flow) trong df.

    Tra ve:
    - edge_index: mang shape (2, so_canh) - hang 0 la id dinh nguon, hang 1 la id dinh dich
    - edge_attr: mang shape (so_canh, so_dac_trung) - dac trung cua tung canh (da chuan hoa o Giai doan 1)
    """
    src_idx = src_key.map(node_id).to_numpy()
    dst_idx = dst_key.map(node_id).to_numpy()
    edge_index = np.vstack([src_idx, dst_idx]).astype("int64")

    edge_attr = df[feature_cols].to_numpy(dtype="float32")

    return edge_index, edge_attr
