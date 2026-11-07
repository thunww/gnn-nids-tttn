"""Phase 01 -- protocol du lieu dung chung cho RF / XGBoost / GNN.

Nguyen tac:
- Moi flow co mot flow_id on dinh (= vi tri dong trong CSV goc, tinh TRUOC khi loc).
- MOT split duy nhat theo KHOI LIEN TUC theo thu tu nguon (V2 khong co timestamp tin cay nen
  KHONG goi la time-based split), co "purge gap" giua cac khoi.
- Clip + scaler chi fit tren flow train. Giu ban raw (chua clip) cua test.
- Nhan theo task (binary / multiclass) co mapping khoa va benign_class_id ro rang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from etl.config import ATTACK_COL, ATTACK_ENCODED_COL, IDENTIFIER_COLS, LABEL_COL

FLOW_ID_COL = "flow_id"
SPLIT_COL = "split"
SPLITS = ("train", "val", "test")
PURGED = "purged"
TASKS = ("binary", "multiclass")
TARGET_COLS = {"binary": "target_binary", "multiclass": "target_multiclass"}
BINARY_CLASS_NAMES = {0: "Benign", 1: "Attack"}

NON_FEATURE_COLS = IDENTIFIER_COLS + [
    LABEL_COL,
    ATTACK_COL,
    ATTACK_ENCODED_COL,
    FLOW_ID_COL,
    SPLIT_COL,
    *TARGET_COLS.values(),
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Cot dac trung theo DUNG thu tu trong df; loai moi cot dinh danh/nhan/metadata."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# --------------------------------------------------------------------------- split


def block_split(
    n: int,
    gap: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> np.ndarray:
    """Gan nhan split cho n flow da sap theo thu tu nguon.

    Chia 3 khoi lien tuc theo ty le, bo `gap` flow o moi ranh gioi (nhan "purged") de:
    (a) session/flow keo dai qua ranh gioi khong roi vao 2 tap, (b) khong co cua so nao
    cat ngang ranh gioi. Cua so duoc dung RIENG trong tung khoi nen khong flow nao xuat hien
    o hai tap.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios phai co tong bang 1")
    if gap < 0:
        raise ValueError("gap phai >= 0")

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    labels = np.full(n, PURGED, dtype=object)
    val_start = n_train + gap
    val_end = val_start + n_val
    test_start = val_end + gap

    labels[:n_train] = "train"
    if val_start < val_end <= n:
        labels[val_start:val_end] = "val"
    if test_start < n:
        labels[test_start:] = "test"
    return labels


# --------------------------------------------------------------------------- nhan


def build_label_mapping(attack_series: pd.Series, task: str) -> dict:
    """Mapping khoa cho 1 task. Duoc dung tu TEN nhan (khong dung thong ke split).

    multiclass: id theo thu tu alphabet cua `Attack` (giu nhu LabelEncoder cu, nen CSE-CIC
    Benign=0, UNSW Benign=2). binary: 0=Benign, 1=Attack.
    Tra ve: {"task", "class_names": {id: ten}, "benign_class_id", "num_classes",
             "attack_to_id": {ten_goc: id}}
    """
    if task not in TASKS:
        raise ValueError(f"task phai thuoc {TASKS}")
    names = sorted(str(v) for v in pd.unique(attack_series.dropna()))
    if "Benign" not in names:
        raise ValueError("Khong tim thay nhan 'Benign' trong cot Attack")

    if task == "multiclass":
        attack_to_id = {name: i for i, name in enumerate(names)}
        class_names = {i: name for name, i in attack_to_id.items()}
        benign_id = attack_to_id["Benign"]
    else:
        attack_to_id = {name: (0 if name == "Benign" else 1) for name in names}
        class_names = dict(BINARY_CLASS_NAMES)
        benign_id = 0

    return {
        "task": task,
        "class_names": {int(k): v for k, v in class_names.items()},
        "benign_class_id": int(benign_id),
        "num_classes": len(class_names),
        "attack_to_id": attack_to_id,
    }


def encode_target(attack_series: pd.Series, mapping: dict) -> np.ndarray:
    """Doi cot Attack -> id lop theo mapping da khoa; bao loi neu gap nhan la."""
    ids = attack_series.astype(str).map(mapping["attack_to_id"])
    if ids.isna().any():
        unknown = sorted(set(attack_series.astype(str)[ids.isna()]))
        raise ValueError(f"Nhan khong co trong mapping: {unknown}")
    return ids.to_numpy(dtype="int64")


# --------------------------------------------------------------------------- tien xu ly


def fit_preprocessor(
    train: pd.DataFrame, feature_cols: list[str], clip_quantile: float = 0.99
) -> tuple[StandardScaler, pd.Series]:
    """Fit nguong clip + StandardScaler CHI tren flow train. Khong sua `train`."""
    upper_bound = train[feature_cols].quantile(clip_quantile)
    clipped = train[feature_cols].clip(upper=upper_bound, axis=1)
    scaler = StandardScaler().fit(clipped)
    return scaler, upper_bound


def transform(
    df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler, upper_bound: pd.Series
) -> pd.DataFrame:
    """Ap dung clip + scaler da fit len ban sao cua df (khong fit lai)."""
    out = df.copy()
    out[feature_cols] = out[feature_cols].clip(upper=upper_bound, axis=1)
    out[feature_cols] = scaler.transform(out[feature_cols]).astype("float32")
    return out


# --------------------------------------------------------------------------- audit


def check_disjoint_splits(ids_by_split: dict[str, np.ndarray]) -> dict[str, int]:
    """So flow_id giao nhau giua tung cap split (phai bang 0)."""
    names = list(ids_by_split)
    overlaps = {}
    sets = {k: set(np.asarray(v).tolist()) for k, v in ids_by_split.items()}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}&{b}"] = len(sets[a] & sets[b])
    return overlaps


def assert_disjoint_splits(ids_by_split: dict[str, np.ndarray]) -> None:
    bad = {k: v for k, v in check_disjoint_splits(ids_by_split).items() if v}
    if bad:
        raise AssertionError(f"Ro ri du lieu: flow_id xuat hien o nhieu split: {bad}")


def assert_feature_order(columns: list[str], expected: list[str]) -> None:
    if list(columns) != list(expected):
        raise AssertionError("Thu tu/tap dac trung khac voi feature contract da luu")


def class_support(df: pd.DataFrame, mapping: dict, split_col: str = SPLIT_COL) -> pd.DataFrame:
    """Bang support: so flow cua tung lop trong tung split (lop vang -> 0)."""
    names = mapping["class_names"]
    table = (
        df.groupby([split_col, TARGET_COLS[mapping["task"]]]).size().unstack(split_col, fill_value=0)
        .reindex(index=sorted(names), fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
    )
    table.insert(0, "class_name", [names[i] for i in table.index])
    table.index.name = "class_id"
    return table
