"""Kiem thu Phase 01: phai BAT DUOC loi co chu y, khong chi kiem tra shape."""

import numpy as np
import pandas as pd
import pytest

from etl.config import IDENTIFIER_COLS
from etl.protocol import (
    FLOW_ID_COL,
    PURGED,
    SPLIT_COL,
    TARGET_COLS,
    assert_disjoint_splits,
    assert_feature_order,
    block_split,
    build_label_mapping,
    class_support,
    encode_target,
    feature_columns,
    fit_preprocessor,
    transform,
)
from graph.build_graph import build_graph
from graph.windowing import sliding_windows


def _flows(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attack = np.array(["Benign"] * n, dtype=object)
    attack[100:140] = "DoS"
    attack[300:320] = "Brute"
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 7}" for i in range(n)],
            "L4_SRC_PORT": rng.integers(1000, 1100, n),
            "IPV4_DST_ADDR": [f"10.0.1.{i % 5}" for i in range(n)],
            "L4_DST_PORT": rng.choice([21, 22, 80], n),
            "F_A": rng.normal(10, 2, n),
            "F_B": rng.normal(0, 1, n),
            "Label": (attack != "Benign").astype(int),
            "Attack": attack,
        }
    )
    df.insert(0, FLOW_ID_COL, np.arange(n))
    return df


def test_block_split_disjoint_with_purge_gap():
    labels = block_split(1000, gap=50)
    assert (labels == PURGED).sum() == 100  # 2 ranh gioi x 50
    idx = {s: np.where(labels == s)[0] for s in ("train", "val", "test")}
    assert idx["train"].max() + 50 < idx["val"].min()
    assert idx["val"].max() + 50 < idx["test"].min()
    assert_disjoint_splits(idx)


def test_leakage_detector_catches_overlapping_split():
    ids = {"train": np.arange(0, 100), "test": np.arange(90, 150)}  # co y giao 10 flow
    with pytest.raises(AssertionError):
        assert_disjoint_splits(ids)


def test_windows_inside_split_never_share_flows_across_splits():
    df = _flows(1000)
    df[SPLIT_COL] = block_split(len(df), gap=40)
    seen = {}
    for s in ("train", "val", "test"):
        part = df[df[SPLIT_COL] == s].reset_index(drop=True)
        ids = set()
        for w in sliding_windows(part, 40, 0.5):
            ids |= set(w[FLOW_ID_COL])
        seen[s] = np.array(sorted(ids))
    assert_disjoint_splits(seen)


def test_scaler_and_clip_fit_on_train_only():
    df = _flows(400)
    train, test = df.iloc[:300].copy(), df.iloc[300:].copy()
    test.loc[test.index[0], "F_A"] = 1e9  # outlier chi co o test
    feats = ["F_A", "F_B"]
    scaler, ub = fit_preprocessor(train, feats)
    # tham so phai hoan toan quyet dinh boi train
    assert ub["F_A"] == pytest.approx(train["F_A"].quantile(0.99))
    assert scaler.n_samples_seen_ == len(train)
    assert scaler.mean_[0] < 20  # khong bi outlier test keo len
    out = transform(test, feats, scaler, ub)
    assert out["F_A"].max() < 50  # outlier da bi clip theo nguong train


def test_label_mapping_keeps_benign_id_not_zero():
    # kieu UNSW: sap alphabet -> Benign khong phai id 0
    s = pd.Series(["Analysis", "Backdoor", "Benign", "DoS", "Benign"])
    m = build_label_mapping(s, "multiclass")
    assert m["benign_class_id"] == 2
    assert m["num_classes"] == 4
    b = build_label_mapping(s, "binary")
    assert b["benign_class_id"] == 0 and b["num_classes"] == 2
    assert list(encode_target(s, b)) == [1, 1, 0, 1, 0]


def test_encode_target_rejects_unknown_label():
    m = build_label_mapping(pd.Series(["Benign", "DoS"]), "multiclass")
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Benign", "Recon"]), m)


def test_graph_edge_order_matches_flow_ids_and_labels():
    df = _flows(400)
    mapping = build_label_mapping(df["Attack"], "multiclass")
    df[TARGET_COLS["multiclass"]] = encode_target(df["Attack"], mapping)
    feats = ["F_A", "F_B"]
    window = df.iloc[90:150].reset_index(drop=True)
    g = build_graph(window, feats, target_col=TARGET_COLS["multiclass"], flow_id_col=FLOW_ID_COL)

    assert g.flow_id.tolist() == window[FLOW_ID_COL].tolist()
    assert g.y.tolist() == window[TARGET_COLS["multiclass"]].tolist()
    # canh i phai mang dung dac trung cua flow i
    assert np.allclose(g.edge_attr.numpy(), window[feats].to_numpy("float32"))
    # tra nguoc nhan that theo flow_id khop y
    truth = df.set_index(FLOW_ID_COL)[TARGET_COLS["multiclass"]]
    assert truth.loc[g.flow_id.numpy()].tolist() == g.y.tolist()


def test_build_graph_backward_compatible_without_flow_id():
    df = _flows(50)
    g = build_graph(df, ["F_A", "F_B"])  # kieu goi cu cua infer.py / pipeline thuc tap
    assert not hasattr(g, "flow_id") or g.get("flow_id") is None
    assert g.y.tolist() == df["Label"].tolist()


def test_feature_order_swap_is_detected():
    df = _flows(10)
    feats = feature_columns(df)
    assert feats == ["F_A", "F_B"]
    with pytest.raises(AssertionError):
        assert_feature_order(list(reversed(feats)), feats)


def test_class_support_reports_absent_class_as_zero():
    df = _flows(1000)
    df[SPLIT_COL] = block_split(len(df), gap=10)
    m = build_label_mapping(df["Attack"], "multiclass")
    df[TARGET_COLS["multiclass"]] = encode_target(df["Attack"], m)
    # chi so 1000: DoS (100-140) va Brute (300-320) nam trong train -> val/test phai bang 0
    t = class_support(df, m)
    dos = t[t["class_name"] == "DoS"].iloc[0]
    assert dos["train"] == 40 and dos["val"] == 0 and dos["test"] == 0


def test_cross_split_overlap_detects_identical_rows_and_ip_pairs():
    from etl.protocol import cross_split_overlap

    df = _flows(100)
    df[SPLIT_COL] = ["train"] * 60 + ["val"] * 20 + ["test"] * 20
    df.loc[80:99, ["F_A", "F_B"]] = df.loc[0:19, ["F_A", "F_B"]].to_numpy()  # test = ban sao train
    out = cross_split_overlap(df, ["F_A", "F_B"])
    assert out["test"]["identical_feature_rows"] == 20
    assert out["val"]["identical_feature_rows"] == 0
    assert out["test"]["ip_pair_seen_in_train_rate"] == 1.0


def test_fit_preprocessor_equals_sklearn_reference():
    from sklearn.preprocessing import StandardScaler

    df = _flows(500, seed=3)
    df.loc[5, "F_A"] = 1e6
    feats = ["F_A", "F_B"]
    mask = np.arange(len(df)) < 350
    scaler, ub = fit_preprocessor(df, feats, mask=mask)

    train = df[mask]
    ref_ub = train[feats].quantile(0.99)
    ref = StandardScaler().fit(train[feats].clip(upper=ref_ub, axis=1))
    assert np.allclose(ub.to_numpy(), ref_ub.to_numpy())
    assert np.allclose(scaler.mean_, ref.mean_) and np.allclose(scaler.scale_, ref.scale_)

    got = transform(df, feats, scaler, ub)[feats].to_numpy()
    want = ref.transform(df[feats].clip(upper=ref_ub, axis=1))
    assert np.allclose(got, want, atol=1e-4)


def test_read_raw_matches_plain_read_csv(tmp_path):
    from etl.run_etl_v2 import read_raw

    n = 50
    raw = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 4}" for i in range(n)],
            "L4_SRC_PORT": np.arange(n) + 1000,
            "IPV4_DST_ADDR": [f"10.0.1.{i % 3}" for i in range(n)],
            "L4_DST_PORT": [80] * n,
            "F_A": np.linspace(0, 5, n),
            "F_B": np.arange(n) * 3,
            "Label": [0] * 40 + [1] * 10,
            "Attack": ["Benign"] * 40 + ["DoS"] * 10,
        }
    )
    path = tmp_path / "x.csv"
    raw.to_csv(path, index=False)
    got = read_raw(path, None, chunk_rows=7)  # chunk nho de thu ghep nhieu khoi
    assert list(got.columns) == list(raw.columns)
    assert got["IPV4_SRC_ADDR"].astype(str).tolist() == raw["IPV4_SRC_ADDR"].tolist()
    assert got["Attack"].astype(str).tolist() == raw["Attack"].tolist()
    assert np.allclose(got[["F_A", "F_B"]].to_numpy(), raw[["F_A", "F_B"]].to_numpy())
    assert got["L4_SRC_PORT"].tolist() == raw["L4_SRC_PORT"].tolist()
    assert len(read_raw(path, 20, chunk_rows=7)) == 20
