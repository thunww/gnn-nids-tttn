import csv
import sys
import warnings
from pathlib import Path

# Cho phep chay truc tiep "python src/models/evaluate_cross_dataset.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# sklearn canh bao "X does not have valid feature names" moi lan goi scaler.transform() tren
# mang numpy thuan (khong co ten cot) -- rescale_graph() goi ham nay lap lai cho tung do thi
# (hang nghin lan tren tap test), gay ngap console. Vo hai (gia tri tinh ra van dung), chi tat
# cho de doc log.
warnings.filterwarnings("ignore", message=".*valid feature names.*", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from etl.config import DATASETS
from etl.scale import apply_scale
from graph.node_features import compute_node_features
from models.config import DEFAULT_PROCESSED_DIR, NON_FEATURE_COLS, TARGET_COL
from models.gnn_config import EDGE_FEATURE_DIM, HIDDEN_DIM, HIDDEN_DIM_BY_DATASET, NODE_FEATURE_DIM, NUM_LAYERS
from models.graphsage import GraphSAGEEdgeClassifier
from models.metrics import compute_full_metrics

# Thi nghiem 2 (TN2, docs/00_research_plan.md muc 6.3): danh gia model da train tren bo
# NGUON, KHONG train/tinh chinh gi them, chay thang tren tap test cua bo DICH -- kiem tra
# kha nang tong quat hoa thuc su sang moi truong mang khac. Ap lai dung scaler+upper_bound
# cua bo NGUON len du lieu bo DICH (thay vi de bo dich tu scale rieng theo thong ke cua no)
# de tach dung "khac biet moi truong that" khoi "nhieu do lech cong thuc scale" -- xem
# docs/decisions.md muc 2026-07-27.


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def rescale_to_source(
    target_test: pd.DataFrame, feature_cols: list[str], target_dir: Path, source_dir: Path
) -> pd.DataFrame:
    """Dao nguoc scale cua bo DICH (lay lai gia tri gan-tho), roi ap lai dung scaler +
    upper_bound cua bo NGUON -- mo phong dung tinh huong "dem model da train di trien khai
    o moi truong moi, khong co san thong ke cua moi truong do".
    """
    target_scaler = joblib.load(target_dir / "scaler.joblib")
    raw_like = target_test.copy()
    raw_like[feature_cols] = target_scaler.inverse_transform(target_test[feature_cols])

    source_scaler = joblib.load(source_dir / "scaler.joblib")
    source_upper_bound = joblib.load(source_dir / "upper_bound.joblib")
    return apply_scale(raw_like, feature_cols, source_scaler, source_upper_bound)


def evaluate_baseline_cross(processed_dir: Path, source: str, target: str, model_name: str) -> dict | None:
    model_path = processed_dir / source / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"  BO QUA: chua co model tai {model_path}")
        return None

    target_test = pd.read_parquet(processed_dir / target / "test.parquet")
    feature_cols = feature_columns(target_test)
    rescaled = rescale_to_source(target_test, feature_cols, processed_dir / target, processed_dir / source)

    model = joblib.load(model_path)
    X_test, y_test = rescaled[feature_cols], rescaled[TARGET_COL]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_full_metrics(y_test, y_pred, y_proba)


def rescale_graph(data: Data, target_dir: Path, source_dir: Path) -> Data:
    """Doi thang do edge_attr cua 1 do thi tu bo DICH sang thang do bo NGUON, roi TINH LAI
    dac trung node (x) -- vi x tong hop (trung binh) tu edge_attr, doi thang do edge_attr
    ma khong tinh lai x se lam sai lech (x cu van con theo thang do cu).
    """
    target_scaler = joblib.load(target_dir / "scaler.joblib")
    source_scaler = joblib.load(source_dir / "scaler.joblib")
    source_upper_bound = joblib.load(source_dir / "upper_bound.joblib").to_numpy()

    edge_attr_raw = target_scaler.inverse_transform(data.edge_attr.numpy())
    edge_attr_clipped = np.clip(edge_attr_raw, a_min=None, a_max=source_upper_bound)
    edge_attr_rescaled = source_scaler.transform(edge_attr_clipped).astype("float32")

    edge_index_np = data.edge_index.numpy()
    x_new = compute_node_features(edge_index_np, edge_attr_rescaled, data.x.shape[0])

    return Data(
        x=torch.tensor(x_new, dtype=torch.float32),
        edge_index=data.edge_index,
        edge_attr=torch.tensor(edge_attr_rescaled, dtype=torch.float32),
        y=data.y,
    )


def evaluate_graphsage_cross(processed_dir: Path, source: str, target: str, device: torch.device) -> dict | None:
    model_path = processed_dir / source / "models" / "graphsage_best.pt"
    if not model_path.exists():
        print(f"  BO QUA: chua co model tai {model_path}")
        return None

    target_graphs = torch.load(processed_dir / target / "test_graphs.pt", weights_only=False)
    print(f"  dang tinh lai dac trung cho {len(target_graphs)} do thi theo thang do cua {source}...")
    rescaled_graphs = [rescale_graph(g, processed_dir / target, processed_dir / source) for g in target_graphs]

    hidden_dim = HIDDEN_DIM_BY_DATASET.get(source, HIDDEN_DIM)
    model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    loader = DataLoader(rescaled_graphs, batch_size=32)
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            proba = F.softmax(out, dim=1)[:, 1]
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            all_proba.extend(proba.cpu().tolist())

    return compute_full_metrics(all_labels, all_preds, all_proba)


def run(processed_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_names = list(DATASETS)
    rows = []

    for source in folder_names:
        for target in folder_names:
            if source == target:
                continue  # doi chung (train=test cung bo) da co o Thi nghiem 1, khong lap lai
            print(f"=== train tren {source}, test tren {target} ===")
            for model_name, evaluator in [
                ("random_forest", lambda s=source, t=target: evaluate_baseline_cross(processed_dir, s, t, "random_forest")),
                ("xgboost", lambda s=source, t=target: evaluate_baseline_cross(processed_dir, s, t, "xgboost")),
                ("graphsage", lambda s=source, t=target: evaluate_graphsage_cross(processed_dir, s, t, device)),
            ]:
                print(f"--- {model_name} ---")
                metrics = evaluator()
                if metrics is None:
                    continue
                print(
                    f"  accuracy={metrics['accuracy']:.4f}  precision_macro={metrics['precision_macro']:.4f}"
                    f"  recall_macro={metrics['recall_macro']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
                    f"  auc_roc={metrics['auc_roc']:.4f}  mcc={metrics['mcc']:.4f}"
                )
                rows.append({"train_on": source, "test_on": target, "model": model_name, **metrics})

    out_path = processed_dir / "cross_dataset_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDa luu bang ket qua day du tai: {out_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
