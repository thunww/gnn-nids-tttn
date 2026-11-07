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
    names = sorted(str(v) for v in attack_series.dropna().unique())
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


def load_mapping(contract: dict, task: str) -> dict:
    """Lay mapping ttu contract.json va doi khoa class_names ve int (JSON luu khoa la chuoi)."""
    m = dict(contract["label_mappings"][task])
    m["class_names"] = {int(k): v for k, v in m["class_names"].items()}
    return m


def encode_target(attack_series: pd.Series, mapping: dict) -> np.ndarray:
    """Doi cot Attack -> id lop theo mapping da khoa; bao loi neu gap nhan la.
    Dung duong tat theo `categories` khi cot la category (tiet kiem RAM voi hang chuc trieu dong)."""
    table = mapping["attack_to_id"]
    if isinstance(attack_series.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in attack_series.cat.categories]
        unknown = [c for c in cats if c not in table]
        if unknown and (attack_series.isin(attack_series.cat.categories[[cats.index(u) for u in unknown]])).any():
            raise ValueError(f"Nhan khong co trong mapping: {sorted(unknown)}")
        lut = np.array([table.get(c, -1) for c in cats], dtype="int64")
        return lut[attack_series.cat.codes.to_numpy()]
    ids = attack_series.astype(str).map(table)
    if ids.isna().any():
        unknown = sorted(set(attack_series.astype(str)[ids.isna()]))
        raise ValueError(f"Nhan khong co trong mapping: {unknown}")
    return ids.to_numpy(dtype="int64")


# --------------------------------------------------------------------------- tien xu ly


def fit_preprocessor(
    train: pd.DataFrame, feature_cols: list[str], clip_quantile: float = 0.99, mask=None
) -> tuple[StandardScaler, pd.Series]:
    """Fit nguong clip + StandardScaler CHI tren flow train, tinh TUNG COT de khong nhan doi RAM
    (ket qua tuong duong StandardScaler().fit(train.clip(upper=quantile)))."""
    n = int(mask.sum()) if mask is not None else len(train)
    ub, mean, var = {}, np.empty(len(feature_cols)), np.empty(len(feature_cols))
    for j, c in enumerate(feature_cols):
        col = (train[c] if mask is None else train.loc[mask, c]).to_numpy(dtype="float64")
        ub[c] = np.quantile(col, clip_quantile)
        col = np.minimum(col, ub[c])
        mean[j], var[j] = col.mean(), col.var()
        del col
    scaler = StandardScaler()
    scaler.mean_, scaler.var_ = mean, var
    scale = np.sqrt(var)
    scale[scale == 0.0] = 1.0  # cung quy uoc voi sklearn cho cot hang so
    scaler.scale_ = scale
    scaler.n_samples_seen_ = np.int64(n)
    scaler.n_features_in_ = len(feature_cols)
    scaler.feature_names_in_ = np.asarray(feature_cols, dtype=object)
    return scaler, pd.Series(ub, index=feature_cols)


def transform(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    upper_bound: pd.Series,
    inplace: bool = False,
) -> pd.DataFrame:
    """Ap dung clip + scaler da fit (khong fit lai), tung cot. inplace=True de khong nhan doi RAM."""
    out = df if inplace else df.copy()
    for j, c in enumerate(feature_cols):
        v = np.minimum(out[c].to_numpy(dtype="float64"), float(upper_bound[c]))
        out[c] = ((v - scaler.mean_[j]) / scaler.scale_[j]).astype("float32")
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


def _row_hash(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


def cross_split_overlap(df: pd.DataFrame, feats: list[str], split_col: str = SPLIT_COL) -> dict:
    """Do muc chong lan giua cac split (chi de bao cao, khong tu y xoa mau).

    - identical_feature_rows: so flow val/test co vector dac trung GIONG HET 1 flow train
      (flow lap lai hop le nhu DoS van tinh, nhung can bao cao de doc so lieu cho dung).
    - ip_pair_seen_in_train: ty le flow val/test co cap (IP nguon, IP dich) da xuat hien o train
      (xap xi muc "phien" trung -- V2 khong co session id).
    """
    h = _row_hash(df, feats)
    pair = _row_hash(df, ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
    split = df[split_col].to_numpy()
    tr = split == "train"
    out = {}
    for s in ("val", "test"):
        m = split == s
        n = int(m.sum())
        dup = np.isin(h[m], h[tr])
        out[s] = {
            "flows": n,
            "identical_feature_rows": int(dup.sum()),
            "identical_feature_rate": float(dup.mean()) if n else 0.0,
            "ip_pair_seen_in_train_rate": float(np.isin(pair[m], pair[tr]).mean()) if n else 0.0,
        }
    return out


def novel_flow_ids(df: pd.DataFrame, feats: list[str], split_col: str = SPLIT_COL) -> pd.DataFrame:
    """flow_id cua cac flow val/test co vector dac trung KHONG xuat hien o train.

    Dung de bao cao them metric tren tap "flow moi" -- phan tach kha nang nho thuoc long
    (flow lap lai) khoi kha nang tong quat hoa that. Khong thay the test day du.
    """
    h = _row_hash(df, feats)
    split = df[split_col].to_numpy()
    seen = np.isin(h, h[split == "train"])
    mask = np.isin(split, ["val", "test"]) & ~seen
    return df.loc[mask, [FLOW_ID_COL, split_col]].reset_index(drop=True)
