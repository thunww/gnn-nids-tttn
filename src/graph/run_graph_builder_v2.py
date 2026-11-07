"""Phase 01: dung do thi TRONG TUNG split (cua so khong bao gio cat ngang ranh gioi), ghi shard
tang dan (RAM bi chan), luu flow_id theo dung thu tu canh.

    python src/graph/run_graph_builder_v2.py --dataset nf-unsw-nb15-v2 --task multiclass \
        --processed data/processed_v2_sample
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from etl.config import DATASETS
from etl.protocol import FLOW_ID_COL, SPLIT_COL, SPLITS, TARGET_COLS, assert_feature_order, feature_columns
from graph.build_graph import build_graph
from graph.config import WINDOW_OVERLAP, WINDOW_SIZE_BY_DATASET
from graph.windowing import sliding_windows

SHARD_SIZE = 256  # so do thi / shard -- cung la so cua so nam trong RAM cung luc


def _build_one(args):
    window, feats, target_col = args
    return build_graph(window, feats, target_col=target_col, flow_id_col=FLOW_ID_COL)


def build_split(df_split: pd.DataFrame, feats: list[str], target_col: str, window_size: int,
                out_dir: Path, split: str, pool: Pool | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{split}_shard_*.pt"):
        old.unlink()

    n_graphs = n_shards = 0
    batch = []

    def flush():
        nonlocal batch, n_graphs, n_shards
        if not batch:
            return
        args = [(w, feats, target_col) for w in batch]
        graphs = pool.map(_build_one, args, chunksize=4) if pool else [_build_one(a) for a in args]
        torch.save(graphs, out_dir / f"{split}_shard_{n_shards:04d}.pt")
        n_graphs += len(graphs)
        n_shards += 1
        batch = []

    for window in sliding_windows(df_split, window_size, WINDOW_OVERLAP):
        batch.append(window)
        if len(batch) >= SHARD_SIZE:
            flush()
    flush()
    return {"graphs": n_graphs, "shards": n_shards, "flows": len(df_split)}


def run(dataset: str, task: str, processed_root: Path, workers: int = 0) -> dict:
    ds_dir = processed_root / dataset
    contract = json.loads((ds_dir / "contract.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(ds_dir / "flows.parquet")

    feats = feature_columns(df)
    assert_feature_order(feats, contract["feature_cols"])
    target_col = TARGET_COLS[task]
    window_size = WINDOW_SIZE_BY_DATASET[dataset]

    out_dir = ds_dir / "graphs" / task
    pool = Pool(workers) if workers > 1 else None
    meta = {
        "dataset": dataset, "task": task, "target_col": target_col,
        "window_size": window_size, "overlap": WINDOW_OVERLAP,
        "feature_hash": contract["feature_hash"],
        "label_mapping": contract["label_mappings"][task],
        "splits": {},
    }
    try:
        for split in SPLITS:
            part = df[df[SPLIT_COL] == split].reset_index(drop=True)
            meta["splits"][split] = build_split(part, feats, target_col, window_size, out_dir, split, pool)
            print(f"  {split}: {meta['splits'][split]}")
    finally:
        if pool:
            pool.close()
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--task", choices=list(TARGET_COLS), required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v2"))
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    run(a.dataset, a.task, a.processed, a.workers)
