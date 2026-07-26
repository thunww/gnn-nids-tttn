import csv
import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/evaluate_test.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from etl.config import DATASETS
from models.config import DEFAULT_PROCESSED_DIR, NON_FEATURE_COLS, TARGET_COL
from models.gnn_config import (
    BATCH_SIZE,
    EDGE_FEATURE_DIM,
    HIDDEN_DIM,
    HIDDEN_DIM_BY_DATASET,
    NODE_FEATURE_DIM,
    NUM_LAYERS,
)
from models.graphsage import GraphSAGEEdgeClassifier
from models.metrics import compute_full_metrics

# CHI chay 1 lan duy nhat cho ket qua chinh thuc (docs/00_research_plan.md muc 4.1) -- script
# nay CHI doc model + tap test da co san, KHONG train/tinh chinh gi.


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def evaluate_baseline(processed_dir: Path, folder_name: str, model_name: str) -> dict | None:
    model_path = processed_dir / folder_name / "models" / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"  BO QUA: chua co model tai {model_path}")
        return None

    model = joblib.load(model_path)
    test = pd.read_parquet(processed_dir / folder_name / "test.parquet")
    feature_cols = feature_columns(test)
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_full_metrics(y_test, y_pred, y_proba)


def evaluate_graphsage(processed_dir: Path, folder_name: str, device: torch.device) -> dict | None:
    model_path = processed_dir / folder_name / "models" / "graphsage_best.pt"
    if not model_path.exists():
        print(f"  BO QUA: chua co model tai {model_path}")
        return None

    test_graphs = torch.load(processed_dir / folder_name / "test_graphs.pt", weights_only=False)
    hidden_dim = HIDDEN_DIM_BY_DATASET.get(folder_name, HIDDEN_DIM)
    model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
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
    rows = []

    for folder_name in DATASETS:
        print(f"=== {folder_name} ===")
        for model_name, evaluator in [
            ("random_forest", lambda: evaluate_baseline(processed_dir, folder_name, "random_forest")),
            ("xgboost", lambda: evaluate_baseline(processed_dir, folder_name, "xgboost")),
            ("graphsage", lambda: evaluate_graphsage(processed_dir, folder_name, device)),
        ]:
            print(f"--- {folder_name} / {model_name} ---")
            metrics = evaluator()
            if metrics is None:
                continue
            print(
                f"  accuracy={metrics['accuracy']:.4f}  precision_macro={metrics['precision_macro']:.4f}"
                f"  recall_macro={metrics['recall_macro']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
                f"  auc_roc={metrics['auc_roc']:.4f}  mcc={metrics['mcc']:.4f}"
            )
            rows.append({"dataset": folder_name, "model": model_name, **metrics})

    out_path = processed_dir / "test_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDa luu bang ket qua day du tai: {out_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
