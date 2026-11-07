"""Phase 01 ETL: split chung + preprocessing train-only + flow_id + nhan theo task.

Khong ghi de artifact thuc tap: dau ra o data/processed_v2/<dataset>/.

Chay mau nho truoc (khuyen nghi, khong ton RAM):
    python src/etl/run_etl_v2.py --dataset nf-unsw-nb15-v2 --max-rows 200000 --out data/processed_v2_sample
Chay day du:
    python src/etl/run_etl_v2.py --dataset nf-cse-cic-ids2018-v2
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from etl.config import ATTACK_COL, DATASETS, DEFAULT_RAW_DIR, LABEL_COL
from etl.protocol import (
    FLOW_ID_COL,
    PURGED,
    SPLIT_COL,
    SPLITS,
    TARGET_COLS,
    assert_disjoint_splits,
    block_split,
    build_label_mapping,
    class_support,
    encode_target,
    feature_columns,
    fit_preprocessor,
    transform,
)
from graph.config import WINDOW_SIZE_BY_DATASET

DEFAULT_OUT = Path("data/processed_v2")


def run(dataset: str, raw_dir: Path, out_root: Path, max_rows: int | None = None) -> dict:
    filename = DATASETS[dataset]
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_dir / dataset / filename, nrows=max_rows)
    n_raw = len(df)
    df[ATTACK_COL] = df[ATTACK_COL].astype(str)

    # flow_id = vi tri dong trong CSV goc, tinh TRUOC khi loc -> moi dong bi bo deu truy vet duoc.
    df.insert(0, FLOW_ID_COL, range(n_raw))
    dedup_cols = [c for c in df.columns if c != FLOW_ID_COL]
    n_dup = int(df.duplicated(subset=dedup_cols).sum())
    df = df.drop_duplicates(subset=dedup_cols)
    n_na = int(df.isna().any(axis=1).sum())
    df = df.dropna().reset_index(drop=True)
    print(f"[{dataset}] raw={n_raw} duplicate={n_dup} missing={n_na} giu={len(df)}")

    # Nhan: mapping khoa tu ten nhan, luu ca 2 task.
    mappings = {t: build_label_mapping(df[ATTACK_COL], t) for t in TARGET_COLS}
    for task, col in TARGET_COLS.items():
        df[col] = encode_target(df[ATTACK_COL], mappings[task])

    feats = feature_columns(df)

    # Split chung theo khoi lien tuc; gap = 1 cua so de khong co phien/cua so nao qua ranh gioi.
    window_size = WINDOW_SIZE_BY_DATASET[dataset]
    df[SPLIT_COL] = block_split(len(df), gap=window_size)
    assert_disjoint_splits({s: df.loc[df[SPLIT_COL] == s, FLOW_ID_COL].to_numpy() for s in SPLITS})
    n_purged = int((df[SPLIT_COL] == PURGED).sum())
    df = df[df[SPLIT_COL] != PURGED].reset_index(drop=True)

    # Preprocessing: fit CHI tren train; giu ban raw cua test (chua clip) cho danh gia cheo sau nay.
    train_mask = df[SPLIT_COL] == "train"
    scaler, upper_bound = fit_preprocessor(df.loc[train_mask], feats)
    df[df[SPLIT_COL] == "test"].to_parquet(out_dir / "raw_test.parquet", index=False)
    df = transform(df, feats, scaler, upper_bound)
    df.to_parquet(out_dir / "flows.parquet", index=False)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(upper_bound, out_dir / "upper_bound.joblib")

    contract = {
        "feature_cols": feats,
        "scaler_fit_rows": int(train_mask.sum()),
        "clip_quantile": 0.99,
        "window_size": window_size,
        "purge_gap": window_size,
        "split_policy": "block theo thu tu nguon (V2 khong co timestamp tin cay)",
        "label_mappings": mappings,
    }
    contract["feature_hash"] = hashlib.sha256("|".join(feats).encode()).hexdigest()[:16]
    (out_dir / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    support = {t: class_support(df, mappings[t]) for t in TARGET_COLS}
    for t, table in support.items():
        table.to_csv(out_dir / f"class_support_{t}.csv")

    audit = {
        "dataset": dataset,
        "raw_rows": n_raw,
        "dropped_duplicate": n_dup,
        "dropped_missing": n_na,
        "purged_at_boundaries": n_purged,
        "rows_per_split": {s: int((df[SPLIT_COL] == s).sum()) for s in SPLITS},
        "flow_id_overlap": "0 (da assert_disjoint_splits)",
        "classes_missing_by_split": {
            t: {s: [int(i) for i in tbl.index[tbl[s] == 0]] for s in SPLITS} for t, tbl in support.items()
        },
    }
    (out_dir / "audit_etl.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-rows", type=int, default=None, help="chi doc N dong dau (chay thu)")
    a = ap.parse_args()
    run(a.dataset, a.raw_dir, a.out, a.max_rows)
