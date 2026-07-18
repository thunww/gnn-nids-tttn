import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/train_baseline.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from etl.config import DATASETS
from models.baselines import build_random_forest, build_xgboost
from models.config import DEFAULT_PROCESSED_DIR, NON_FEATURE_COLS, TARGET_COL


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run(processed_dir: Path) -> None:
    for folder_name in DATASETS:
        dataset_dir = processed_dir / folder_name
        train = pd.read_parquet(dataset_dir / "train.parquet")
        val = pd.read_parquet(dataset_dir / "val.parquet")

        feature_cols = feature_columns(train)
        X_train, y_train = train[feature_cols], train[TARGET_COL]
        X_val, y_val = val[feature_cols], val[TARGET_COL]
        num_classes = y_train.nunique()

        models = {
            "random_forest": build_random_forest(),
            "xgboost": build_xgboost(num_classes),
        }

        out_dir = dataset_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            print(f"=== {folder_name} / {name} ===")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average="macro")
            print(f"  val accuracy={acc:.4f}  val f1_macro={f1_macro:.4f}")

            joblib.dump(model, out_dir / f"{name}.joblib")
            print(f"  luu tai: {out_dir / f'{name}.joblib'}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
