"""Phase 01: RF / XGBoost doc CUNG split chung (flows.parquet, cot `split`) voi GNN.

Mac dinh chi danh gia tren VAL; test chi cham khi truyen --eval-test (chay cuoi, 1 lan).

    python src/models/train_baseline_v2.py --dataset nf-unsw-nb15-v2 --task multiclass
    # chay thu nhanh (ghi ro trong metrics.json):  --max-train-rows 300000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from etl.config import DATASETS
from etl.protocol import FLOW_ID_COL, SPLIT_COL, TARGET_COLS, assert_feature_order, feature_columns, load_mapping
from models.baselines import build_random_forest, build_xgboost

MODEL_BUILDERS = {
    "random_forest": lambda k: build_random_forest(),
    "xgboost": lambda k: build_xgboost(k),
}


def task_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, mapping: dict) -> dict:
    """Metric theo task. Lop vang trong y_true van duoc bao cao (support=0), khong lam crash."""
    labels = sorted(int(k) for k in mapping["class_names"])
    names = [mapping["class_names"][i] for i in labels]
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        # F1-macro chi tinh tren cac lop CO MAT trong y_true (lop support=0 khong dua vao trung binh)
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true.tolist())), zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_roc": float("nan"),
    }
    if proba is not None:
        try:
            if mapping["num_classes"] == 2:
                m["auc_roc"] = roc_auc_score(y_true, proba[:, 1])
            else:
                present = sorted(set(y_true.tolist()))
                p = proba[:, present]
                p = p / p.sum(axis=1, keepdims=True)
                m["auc_roc"] = roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=present)
        except ValueError:
            pass
    m["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
    )
    return {k: (float(v) if not isinstance(v, dict) else v) for k, v in m.items()}


def _full_proba(model, X, num_classes: int) -> np.ndarray:
    """predict_proba ve du num_classes cot (RF chi tra cac lop thay luc train)."""
    p = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.arange(p.shape[1])))
    if p.shape[1] == num_classes and (classes == np.arange(num_classes)).all():
        return p
    full = np.zeros((p.shape[0], num_classes), dtype="float64")
    full[:, classes.astype(int)] = p
    return full


def run(dataset: str, task: str, processed_root: Path, models: list[str], max_train_rows: int | None,
        eval_split: str) -> dict:
    ds_dir = processed_root / dataset
    contract = json.loads((ds_dir / "contract.json").read_text(encoding="utf-8"))
    mapping = load_mapping(contract, task)
    target = TARGET_COLS[task]

    df = pd.read_parquet(ds_dir / "flows.parquet")
    feats = feature_columns(df)
    assert_feature_order(feats, contract["feature_cols"])

    train = df[df[SPLIT_COL] == "train"]
    evals = df[df[SPLIT_COL] == eval_split]
    if max_train_rows and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=42)  # CHI de chay thu, duoc ghi vao metrics
    X_tr, y_tr = train[feats], train[target].to_numpy()
    X_ev, y_ev = evals[feats], evals[target].to_numpy()

    novel_path = ds_dir / "novel_flow_ids.parquet"
    novel_ids = None
    if novel_path.exists():
        nv = pd.read_parquet(novel_path)
        novel_ids = set(nv.loc[nv[SPLIT_COL] == eval_split, FLOW_ID_COL])

    out_dir = ds_dir / "models" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"dataset": dataset, "task": task, "eval_split": eval_split,
               "train_rows_used": len(train), "train_rows_full": int((df[SPLIT_COL] == "train").sum()),
               "subsampled": bool(max_train_rows and len(train) == max_train_rows),
               "eval_rows": len(evals), "feature_hash": contract["feature_hash"], "models": {}}

    for name in models:
        print(f"=== {dataset} / {task} / {name} ===")
        model = MODEL_BUILDERS[name](mapping["num_classes"])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_ev)
        proba = _full_proba(model, X_ev, mapping["num_classes"])
        res = task_metrics(y_ev, np.asarray(pred).astype(int), proba, mapping)
        if novel_ids is not None:
            keep = evals[FLOW_ID_COL].isin(novel_ids).to_numpy()
            res["novel_only"] = {"rows": int(keep.sum())}
            if keep.any():
                nm = task_metrics(y_ev[keep], np.asarray(pred).astype(int)[keep], proba[keep], mapping)
                nm.pop("per_class")
                res["novel_only"].update(nm)
        results["models"][name] = res
        print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)}, "| novel-only f1_macro:", res.get("novel_only", {}).get("f1_macro"))
        joblib.dump(model, out_dir / f"{name}.joblib")

    (out_dir / f"baseline_metrics_{eval_split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--task", choices=list(TARGET_COLS), required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v2"))
    ap.add_argument("--models", nargs="+", choices=list(MODEL_BUILDERS), default=list(MODEL_BUILDERS))
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--eval-test", action="store_true", help="danh gia tren TEST (chi chay cuoi cung)")
    a = ap.parse_args()
    run(a.dataset, a.task, a.processed, a.models, a.max_train_rows, "test" if a.eval_test else "val")
