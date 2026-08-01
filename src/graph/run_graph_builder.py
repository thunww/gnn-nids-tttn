import os
import sys
from multiprocessing import Pool
from pathlib import Path

# Cho phep chay truc tiep "python src/graph/run_graph_builder.py" tu bat ky thu muc nao,
# khong chi khi da them src/ vao sys.path tu truoc (vd trong notebook Colab).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from etl.config import ATTACK_COL, ATTACK_ENCODED_COL, DATASETS, IDENTIFIER_COLS, LABEL_COL
from graph.build_graph import build_graph
from graph.config import DEFAULT_PROCESSED_DIR, WINDOW_OVERLAP, WINDOW_SIZE_BY_DATASET
from graph.windowing import sliding_windows


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, ATTACK_ENCODED_COL]]


def _build_one(args: tuple[pd.DataFrame, list[str]]):
    window, feature_cols = args
    return build_graph(window, feature_cols)


def run(processed_dir: Path, num_workers: int | None = None, folder_names: list[str] | None = None) -> None:
    num_workers = num_workers or min(os.cpu_count() or 4, 8)
    folder_names = folder_names or list(DATASETS)

    for folder_name in folder_names:
        print(f"=== {folder_name} ===")

        # Doc ban GIU NGUYEN THU TU GOC (chua xao/chia) -- moi cua so cat ra la lat cat
        # thoi gian thuc cua luong mang, khong phai cac dong ngau nhien rut tu khap noi.
        path = processed_dir / folder_name / "full_chronological.parquet"
        df = pd.read_parquet(path)
        feature_cols = feature_columns(df)

        window_size = WINDOW_SIZE_BY_DATASET[folder_name]
        print(f"  window_size={window_size}")
        windows = ((window, feature_cols) for window in sliding_windows(df, window_size, WINDOW_OVERLAP))

        with Pool(num_workers) as pool:
            graphs = list(pool.imap(_build_one, windows, chunksize=4))

        print(f"  tong so do thi con: {len(graphs)}")

        # 2026-07-31: SUA LOI RO RI DU LIEU nghiem trong -- xem docs/decisions.md.
        # Cua so lien tiep CHONG LAP 50% (WINDOW_OVERLAP=0.5, xem graph/windowing.py) --
        # cua so i va i+1 dung chung ~50% flow. Truoc day dung train_test_split() xao tron
        # NGAU NHIEN danh sach do thi roi moi chia -- do bang chung: 46.2% cap cua so lien ke
        # bi tach khac tap (vd cua so i vao train, cua so i+1 vao test) -- nghia la cung 1 flow
        # xuat hien o CA train lan test, gay diem so ao cao bat thuong (dac biet ro o TN1).
        #
        # Sua: chia theo KHOI LIEN TUC theo thu tu thoi gian (dung thu tu do thi da dung tu
        # sliding_windows(), KHONG xao tron) -- giu dung nguyen tac da ap dung dung o buoc ETL
        # (full_chronological.parquet). Them "purge gap" 1 do thi o moi ranh gioi de loai bo
        # hoan toan chong lap con sot lai giua 2 do thi ke sat ranh gioi train/val va val/test
        # (chi mat 2 do thi/~18.892, khong dang ke).
        n = len(graphs)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_graphs = graphs[:n_train]
        val_graphs = graphs[n_train + 1 : n_train + 1 + n_val]
        test_graphs = graphs[n_train + 1 + n_val + 1 :]

        for split, split_graphs in [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]:
            out_path = processed_dir / folder_name / f"{split}_graphs.pt"
            torch.save(split_graphs, out_path)
            print(f"  {folder_name}/{split}: {len(split_graphs)} do thi con, luu tai {out_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    # bo dataset thu 3 (tuy chon): chi dung lai 1 bo, tranh dung lai bo khong doi gi
    # (vd chi UNSW-NB15-v2 doi WINDOW_SIZE, CSE-CIC khong can dung lai, do mat rat lau).
    folder_name_filter = sys.argv[2] if len(sys.argv) > 2 else None
    run(processed_dir, folder_names=[folder_name_filter] if folder_name_filter else None)
