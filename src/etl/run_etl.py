import gc
import json
import sys
from pathlib import Path

import joblib

from etl.config import (
    ATTACK_COL,
    DATASETS,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    IDENTIFIER_COLS,
    LABEL_COL,
)
from etl.load import load_raw
from etl.clean import clean
from etl.split import stratified_split
from etl.scale import fit_scale


def run(raw_dir: Path, processed_dir: Path) -> None:
    for folder_name, filename in DATASETS.items():
        print(f"=== {folder_name} ===")

        df = load_raw(raw_dir, folder_name, filename)
        df, attack_mapping = clean(df, ATTACK_COL)

        train, val, test = stratified_split(df, ATTACK_COL)

        feature_cols = [
            c
            for c in df.columns
            if c not in IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, "Attack_encoded"]
        ]
        del df
        gc.collect()

        train, val, test, scaler = fit_scale(train, val, test, feature_cols)

        out_dir = processed_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train.to_parquet(out_dir / "train.parquet", index=False)
        val.to_parquet(out_dir / "val.parquet", index=False)
        test.to_parquet(out_dir / "test.parquet", index=False)

        joblib.dump(scaler, out_dir / "scaler.joblib")
        with open(out_dir / "attack_label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(attack_mapping, f, ensure_ascii=False, indent=2)

        print(f"  train={len(train)} val={len(val)} test={len(test)}")
        print(f"  luu tai: {out_dir}")

        del train, val, test, scaler
        gc.collect()


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW_DIR
    processed_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PROCESSED_DIR
    run(raw_dir, processed_dir)
