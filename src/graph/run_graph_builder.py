import os
import sys
from multiprocessing import Pool
from pathlib import Path

# Cho phep chay truc tiep "python src/graph/run_graph_builder.py" tu bat ky thu muc nao,
# khong chi khi da them src/ vao sys.path tu truoc (vd trong notebook Colab).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from etl.config import ATTACK_COL, ATTACK_ENCODED_COL, DATASETS, IDENTIFIER_COLS, LABEL_COL
from graph.build_graph import build_graph
from graph.config import DEFAULT_PROCESSED_DIR, WINDOW_OVERLAP, WINDOW_SIZE
from graph.windowing import sliding_windows


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, ATTACK_ENCODED_COL]]


def _build_one(args: tuple[pd.DataFrame, list[str]]):
    window, feature_cols = args
    return build_graph(window, feature_cols)


def run(processed_dir: Path, num_workers: int | None = None) -> None:
    num_workers = num_workers or min(os.cpu_count() or 4, 8)

    for folder_name in DATASETS:
        print(f"=== {folder_name} ===")

        # Doc ban GIU NGUYEN THU TU GOC (chua xao/chia) -- moi cua so cat ra la lat cat
        # thoi gian thuc cua luong mang, khong phai cac dong ngau nhien rut tu khap noi.
        path = processed_dir / folder_name / "full_chronological.parquet"
        df = pd.read_parquet(path)
        feature_cols = feature_columns(df)

        windows = ((window, feature_cols) for window in sliding_windows(df, WINDOW_SIZE, WINDOW_OVERLAP))

        with Pool(num_workers) as pool:
            graphs = list(pool.imap(_build_one, windows, chunksize=4))

        print(f"  tong so do thi con: {len(graphs)}")

        # Chia DANH SACH DO THI (khong phai dong) thanh 70/15/15 -- giu nguyen tung do thi
        # da dung tu lat cat thoi gian that, chi xao thu tu GIUA CAC do thi voi nhau.
        train_graphs, temp_graphs = train_test_split(graphs, test_size=0.30, random_state=42)
        val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.50, random_state=42)

        for split, split_graphs in [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]:
            out_path = processed_dir / folder_name / f"{split}_graphs.pt"
            torch.save(split_graphs, out_path)
            print(f"  {folder_name}/{split}: {len(split_graphs)} do thi con, luu tai {out_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
