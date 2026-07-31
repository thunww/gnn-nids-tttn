"""Phase E: nap model GraphSAGE 1 lan luc khoi dong, suy luan cho 1 cua so flow.

Tai su dung dung pattern nap model tu evaluate_graphsage() (src/models/evaluate_test.py) --
KHONG doan lai kien truc/sieu tham so, xem docs/graphsage/05_demo_realtime_setup.md muc 5.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F

from api.zeek_convert import FEATURE_COLS, IDENTIFIER_COLS, convert_rows_to_dataframe, get_scaler_mean
from etl.config import LABEL_COL
from etl.scale import apply_scale
from graph.build_graph import build_graph
from models.gnn_config import EDGE_FEATURE_DIM, HIDDEN_DIM, HIDDEN_DIM_BY_DATASET, NODE_FEATURE_DIM, NUM_LAYERS
from models.graphsage import GraphSAGEEdgeClassifier


class RealtimeDetector:
    """Nap model + scaler 1 lan luc khoi dong server, dung lai cho moi cua so flow (khong nap lai)."""

    def __init__(self, processed_dir: Path, folder_name: str = "nf-cse-cic-ids2018-v2", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model_dir = processed_dir / folder_name
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.upper_bound = joblib.load(model_dir / "upper_bound.joblib")
        self.scaler_mean = get_scaler_mean(self.scaler)

        hidden_dim = HIDDEN_DIM_BY_DATASET.get(folder_name, HIDDEN_DIM)
        self.model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
        state_dict = torch.load(
            model_dir / "models" / "graphsage_best.pt", map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    def predict_window(self, rows: list[dict]) -> pd.DataFrame:
        """rows: list flow tho tu Zeek conn.log (dict, xem zeek_convert.read_conn_log()), dung
        1 cua so hoan chinh (WindowBuffer.add() tra ve).

        Tra ve DataFrame: 4 cot dinh danh + ts/uid goc + pred_label (0=Benign, 1=Attack) +
        pred_proba (xac suat Attack), THEO DUNG THU TU rows dau vao (build_edges giu nguyen
        thu tu dong -> hang out[i] khop dung rows[i]).
        """
        raw_df = convert_rows_to_dataframe(rows, self.scaler_mean)
        scaled_df = apply_scale(raw_df.copy(), FEATURE_COLS, self.scaler, self.upper_bound)
        # build_graph() can cot LABEL_COL de dung Data.y -- khong co nhan that luc live (dien 0,
        # KHONG dung gia tri nay de suy luan, chi de build_graph() khong loi thieu cot).
        scaled_df[LABEL_COL] = 0
        graph = build_graph(scaled_df, FEATURE_COLS).to(self.device)

        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index, graph.edge_attr)
            proba = F.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)

        result = raw_df[IDENTIFIER_COLS].copy()
        result["ts"] = [row.get("ts") for row in rows]
        result["uid"] = [row.get("uid") for row in rows]
        result["pred_label"] = pred.cpu().numpy()
        result["pred_proba"] = proba.cpu().numpy()
        return result
