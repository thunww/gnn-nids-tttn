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
import numpy as np
import pandas as pd

from etl.config import ATTACK_COL, DATASETS, DEFAULT_RAW_DIR, IDENTIFIER_COLS, LABEL_COL
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
    cross_split_overlap,
    encode_target,
    feature_columns,
    fit_preprocessor,
    novel_flow_ids,
    transform,
)
from graph.config import WINDOW_SIZE_BY_DATASET

DEFAULT_OUT = Path("data/processed_v2")


def _count_rows(path: Path) -> int:
    """So dong du lieu (khong tinh header) bang cach dem ky tu xuong dong -- khong nap file vao RAM."""
    n = 0
    with open(path, "rb") as f:
        while block := f.read(1 << 26):
            n += block.count(b"\n")
    return max(n - 1, 0)


def read_raw(path: Path, max_rows: int | None, chunk_rows: int = 1_000_000) -> pd.DataFrame:
    """Doc CSV theo khoi vao mang cap phat TRUOC (dinh RAM ~ kich thuoc bang cuoi, khong nhan doi).

    Dac trung float32 (mot khoi lien tuc), cong int32, Label int8, IP va Attack ma hoa thanh
    category bang bang tra tang dan. Gia tri nguyen > 2^24 bi lam tron o float32 (khong anh huong
    sau khi clip + chuan hoa); viec phat hien dong trung dua tren gia tri float32.
    """
    cols = list(pd.read_csv(path, nrows=0).columns)
    ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    port_cols = ["L4_SRC_PORT", "L4_DST_PORT"]
    feats = [c for c in cols if c not in IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL]]
    n = _count_rows(path)
    if max_rows:
        n = min(n, max_rows)

    feat = np.empty((n, len(feats)), dtype="float32")
    ports = {c: np.empty(n, dtype="int32") for c in port_cols}
    label = np.empty(n, dtype="int8")
    codes = {c: np.empty(n, dtype="int32") for c in ip_cols + [ATTACK_COL]}
    vocab = {c: {} for c in codes}

    # Dac trung doc ra float64 theo tung khoi roi ep ve float32 co kiem soat: vai cot (vd
    # SRC_TO_DST_SECOND_BYTES) co gia tri vo ly ~1e300 trong CSV goc, float32 se tran thanh inf.
    # Bao hoa ve gia tri huu han lon nhat cua float32 (sau do bi clip ve phan vi 99% nhu cu).
    dtype = {c: "float64" for c in feats} | {c: "int32" for c in port_cols} | {LABEL_COL: "int8"}
    f32max = float(np.finfo("float32").max)
    saturated = {}
    pos = 0
    for chunk in pd.read_csv(path, dtype=dtype, chunksize=chunk_rows, nrows=n, low_memory=False):
        m = len(chunk)
        block = chunk[feats].to_numpy(dtype="float64")
        over = np.abs(block) > f32max
        if over.any():
            for j in np.where(over.any(axis=0))[0]:
                saturated[feats[j]] = saturated.get(feats[j], 0) + int(over[:, j].sum())
            block = np.clip(block, -f32max, f32max)
        feat[pos : pos + m] = block.astype("float32")
        for c in port_cols:
            ports[c][pos : pos + m] = chunk[c].to_numpy()
        label[pos : pos + m] = chunk[LABEL_COL].to_numpy()
        for c in codes:
            inv, uniq = pd.factorize(chunk[c].astype(str))
            table = vocab[c]
            glob = np.fromiter((table.setdefault(u, len(table)) for u in uniq), dtype="int32", count=len(uniq))
            codes[c][pos : pos + m] = glob[inv]
        pos += m
        del chunk
    assert pos == n, f"doc {pos} dong, ky vong {n}"
    for c in saturated:
        print(f"  [canh bao] cot {c}: {saturated[c]} gia tri vuot gioi han float32, bao hoa ve max huu han")

    df = pd.DataFrame(feat, columns=feats, copy=False)
    df[LABEL_COL] = label  # thu tu cot cuoi cung giong CSV: ids, dac trung, Label, Attack
    df[ATTACK_COL] = pd.Categorical.from_codes(codes[ATTACK_COL], categories=list(vocab[ATTACK_COL]))
    for i, c in enumerate([ip_cols[0], port_cols[0], ip_cols[1], port_cols[1]]):
        arr = (
            pd.Categorical.from_codes(codes[c], categories=list(vocab[c])) if c in ip_cols else ports[c]
        )
        df.insert(i, c, arr)
    df.attrs["saturated_values"] = saturated
    return df


def _mem(tag: str) -> None:
    """In RSS hien tai (GB) neu cai psutil -- giup do dinh RAM tung buoc tren Colab."""
    try:
        import psutil

        print(f"  [mem] {tag}: {psutil.Process().memory_info().rss / 2**30:.2f} GB")
    except ImportError:
        pass


def run(dataset: str, raw_dir: Path, out_root: Path, max_rows: int | None = None) -> dict:
    filename = DATASETS[dataset]
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_raw(raw_dir / dataset / filename, max_rows)
    n_raw = len(df)
    saturated = dict(df.attrs.get("saturated_values", {}))
    _mem("after_read")

    # flow_id = vi tri dong trong CSV goc, tinh TRUOC khi loc -> moi dong bi bo deu truy vet duoc.
    df.insert(0, FLOW_ID_COL, np.arange(n_raw, dtype="int64"))
    dedup_cols = [c for c in df.columns if c != FLOW_ID_COL]
    # Chi sao chep bang khi that su co dong can bo (tranh nhan doi RAM voi ~19 trieu dong).
    # Phat hien trung bang hash 64-bit tung dong (duplicated() cua pandas giu ma nhom cho ca 47 cot
    # => hang GB voi 19 trieu dong). Xac suat va cham giua 2 dong khac nhau ~ N^2/2^65, bo qua duoc.
    row_hash = pd.util.hash_pandas_object(df[dedup_cols], index=False).to_numpy()
    keep = ~pd.Series(row_hash).duplicated().to_numpy()
    del row_hash
    n_dup = int((~keep).sum())
    if n_dup:
        df = df[keep]
    na_rows = df.isna().any(axis=1).to_numpy()
    n_na = int(na_rows.sum())
    if n_na:
        df = df[~na_rows]
    df = df.reset_index(drop=True)
    _mem("after_clean")
    print(f"[{dataset}] raw={n_raw} duplicate={n_dup} missing={n_na} giu={len(df)}")

    # Nhan: mapping khoa tu ten nhan, luu ca 2 task.
    mappings = {t: build_label_mapping(df[ATTACK_COL], t) for t in TARGET_COLS}
    for task, col in TARGET_COLS.items():
        df[col] = encode_target(df[ATTACK_COL], mappings[task])

    feats = feature_columns(df)

    # Split chung theo khoi lien tuc; gap = 1 cua so de khong co phien/cua so nao qua ranh gioi.
    window_size = WINDOW_SIZE_BY_DATASET[dataset]
    df[SPLIT_COL] = pd.Categorical(block_split(len(df), gap=window_size), categories=[*SPLITS, PURGED])
    assert_disjoint_splits({s: df.loc[df[SPLIT_COL] == s, FLOW_ID_COL].to_numpy() for s in SPLITS})
    # Dong "purged" van nam trong flows.parquet (split="purged") de truy vet va tiet kiem RAM;
    # moi noi dung deu loc theo split in (train, val, test) nen chung khong bao gio duoc dung.
    n_purged = int((df[SPLIT_COL] == PURGED).sum())

    # Preprocessing: fit CHI tren train; giu ban raw cua test (chua clip) cho danh gia cheo sau nay.
    train_mask = df[SPLIT_COL] == "train"
    _mem("after_split")
    scaler, upper_bound = fit_preprocessor(df, feats, mask=train_mask.to_numpy())
    _mem("after_fit")
    df[df[SPLIT_COL] == "test"].to_parquet(out_dir / "raw_test.parquet", index=False)
    df = transform(df, feats, scaler, upper_bound, inplace=True)
    _mem("after_transform")
    df.to_parquet(out_dir / "flows.parquet", index=False)
    _mem("after_parquet")

    novel = novel_flow_ids(df, feats)
    novel.to_parquet(out_dir / "novel_flow_ids.parquet", index=False)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(upper_bound, out_dir / "upper_bound.joblib")

    contract = {
        "feature_cols": feats,
        "scaler_fit_rows": int(train_mask.sum()),
        "clip_quantile": 0.99,
        "window_size": window_size,
        "purge_gap": window_size,
        "purged_rows_kept_in_parquet": True,
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
        "saturated_values_by_column": saturated,
        "dropped_duplicate": n_dup,
        "dropped_missing": n_na,
        "purged_at_boundaries": n_purged,
        "rows_per_split": {s: int((df[SPLIT_COL] == s).sum()) for s in SPLITS},
        "flow_id_overlap": "0 (da assert_disjoint_splits)",
        "cross_split_overlap": cross_split_overlap(df, feats),
        "novel_flows": {s: int((novel[SPLIT_COL] == s).sum()) for s in ("val", "test")},
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
