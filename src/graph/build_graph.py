import pandas as pd
import torch
from torch_geometric.data import Data

from etl.config import ATTACK_ENCODED_COL, IDENTIFIER_COLS
from graph.edges import build_edges
from graph.node_features import compute_node_features
from graph.nodes import build_node_ids


def build_graph(df: pd.DataFrame, feature_cols: list[str]) -> Data:
    """Ghep nodes + edges + node_features thanh 1 object Data cua PyTorch Geometric,
    dung lam dau vao truc tiep cho GCN/GAT o Giai doan 3.
    """
    src_ip_col, src_port_col, dst_ip_col, dst_port_col = IDENTIFIER_COLS

    node_id, src_key, dst_key = build_node_ids(df, src_ip_col, src_port_col, dst_ip_col, dst_port_col)
    edge_index, edge_attr = build_edges(df, node_id, src_key, dst_key, feature_cols)
    x = compute_node_features(edge_index, len(node_id))
    y = df[ATTACK_ENCODED_COL].to_numpy(dtype="int64")  # da lop: loai tan cong cu the, khong phai nhi phan

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
    )
