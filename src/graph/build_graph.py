import pandas as pd
import torch
from torch_geometric.data import Data

from etl.config import IDENTIFIER_COLS, LABEL_COL
from graph.edges import build_edges
from graph.node_features import compute_node_features
from graph.nodes import build_node_ids


def build_graph(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = LABEL_COL,
    flow_id_col: str | None = None,
) -> Data:
    """Ghep nodes + edges + node_features thanh 1 object Data cua PyTorch Geometric,
    dung lam dau vao truc tiep cho GCN/GAT o Giai doan 3.
    """
    src_ip_col, src_port_col, dst_ip_col, dst_port_col = IDENTIFIER_COLS

    node_id, src_key, dst_key = build_node_ids(df, src_ip_col, src_port_col, dst_ip_col, dst_port_col)
    edge_index, edge_attr = build_edges(df, node_id, src_key, dst_key, feature_cols)
    x = compute_node_features(edge_index, edge_attr, len(node_id))
    # nhi phan (2026-07-26, doi tu da lop): xem docs/README.md -- chi con GraphSAGE, uu tien
    # du doan "co tan cong hay khong" thay vi loai tan cong cu the.
    # target_col mac dinh = Label (nhi phan) de khong doi hanh vi cu; Phase 01 truyen
    # target_col theo task (binary/multiclass). flow_id_col (tuy chon) luu ID flow goc theo
    # DUNG thu tu canh de audit chong ro ri va doi chieu du doan voi nhan that.
    y = df[target_col].to_numpy(dtype="int64")

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
    )
    if flow_id_col is not None:
        data.flow_id = torch.tensor(df[flow_id_col].to_numpy(dtype="int64"), dtype=torch.long)
    return data
