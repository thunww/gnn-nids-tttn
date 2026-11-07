import json
import shutil

import numpy as np
import pandas as pd
import torch

from etl.protocol import FLOW_ID_COL, TARGET_COLS, block_split, build_label_mapping, encode_target
from graph.build_graph import build_graph
from graph.windowing import sliding_windows
from models.train_gnn_v2 import (
    class_weights,
    load_bundle,
    parse_args,
    predict_flows,
    shard_paths,
    train,
    unique_train_flow_labels,
)

WINDOW = 40
FEATS = ["F_A", "F_B", "F_C"]


def make_dataset(root, task="multiclass", n=1200, seed=0):
    """Du lieu tong hop co cau truc: lop 'DoS' (F_A cao) va 'Scan' (F_B cao) tach duoc."""
    rng = np.random.default_rng(seed)
    attack = np.array(["Benign"] * n, dtype=object)
    attack[::7] = "DoS"
    attack[3::11] = "Scan"
    df = pd.DataFrame(rng.normal(size=(n, 3)).astype("float32"), columns=FEATS)
    df.loc[attack == "DoS", "F_A"] += 4
    df.loc[attack == "Scan", "F_B"] += 4
    df["IPV4_SRC_ADDR"] = [f"10.0.0.{i % 9}" for i in range(n)]
    df["L4_SRC_PORT"] = rng.integers(1000, 1050, n)
    df["IPV4_DST_ADDR"] = [f"10.0.1.{i % 5}" for i in range(n)]
    df["L4_DST_PORT"] = rng.choice([21, 22, 80], n)
    df["Attack"] = attack
    df[FLOW_ID_COL] = np.arange(n)
    df["split"] = block_split(n, gap=WINDOW)
    mapping = build_label_mapping(df["Attack"], task)
    df[TARGET_COLS[task]] = encode_target(df["Attack"], mapping)

    gdir = root / "ds" / "graphs" / task
    gdir.mkdir(parents=True)
    meta = {"label_mapping": mapping, "feature_hash": "x", "window_size": WINDOW, "splits": {}}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s].reset_index(drop=True)
        graphs = [build_graph(w, FEATS, TARGET_COLS[task], FLOW_ID_COL) for w in sliding_windows(part, WINDOW, 0.5)]
        torch.save(graphs[:8], gdir / f"{s}_shard_0000.pt")
        if len(graphs) > 8:
            torch.save(graphs[8:], gdir / f"{s}_shard_0001.pt")
    (gdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return mapping


def args_for(root, **kw):
    base = ["--dataset", "nf-unsw-nb15-v2", "--task", "multiclass", "--processed", str(root),
            "--hidden-dim", "8", "--batch-size", "4", "--patience", "50"]
    for k, v in kw.items():
        base += [f"--{k.replace('_', '-')}", str(v)]
    return parse_args(base)


def setup_root(tmp_path):
    root = tmp_path / "p"
    make_dataset(root)
    # test dung ten dataset hop le trong DATASETS: doi ten thu muc
    (root / "ds").rename(root / "nf-unsw-nb15-v2")
    return root


def test_class_weights_finite_with_absent_class_and_unique_flows(tmp_path):
    root = setup_root(tmp_path)
    paths = shard_paths(root / "nf-unsw-nb15-v2" / "graphs" / "multiclass", "train")
    labels = unique_train_flow_labels(paths)
    ids = np.concatenate([torch.cat([g.flow_id for g in torch.load(p, weights_only=False)]).numpy() for p in paths])
    assert len(labels) == len(np.unique(ids)) < len(ids)  # cua so chong lap: flow chi tinh 1 lan
    w = class_weights(labels, num_classes=5, loss="cb")  # lop 3, 4 vang -> van huu han
    assert torch.isfinite(w).all() and w.shape == (5,)
    assert class_weights(labels, 3, "ce") is None


def test_predict_flows_dedups_overlapping_windows(tmp_path):
    root = setup_root(tmp_path)
    paths = shard_paths(root / "nf-unsw-nb15-v2" / "graphs" / "multiclass", "val")
    args = args_for(root, epochs=1)
    train(args)
    model, _ = load_bundle(root / "nf-unsw-nb15-v2" / "models" / "multiclass" / "graphsage_cb_seed42" / "best_bundle.pt")
    pred = predict_flows(model, paths, torch.device("cpu"))
    assert pred[FLOW_ID_COL].is_unique
    assert (pred["windows"] == 2).any()  # co flow thuoc 2 cua so
    pcols = [c for c in pred.columns if c.startswith("proba_")]
    assert np.allclose(pred[pcols].sum(axis=1), 1.0, atol=1e-5)


def test_resume_equals_uninterrupted_training(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    make_dataset(root_a)
    shutil.copytree(root_a, root_b)
    for r in (root_a, root_b):
        (r / "ds").rename(r / "nf-unsw-nb15-v2")

    train(args_for(root_a, epochs=3))                 # lien tuc 3 epoch
    train(args_for(root_b, epochs=1))                 # dung o epoch 1 ...
    state = train(args_for(root_b, epochs=3))         # ... roi RESUME den epoch 3
    assert [h["epoch"] for h in state["history"]] == [1, 2, 3]

    ra = torch.load(root_a / "nf-unsw-nb15-v2/models/multiclass/graphsage_cb_seed42/last.pt", weights_only=False)
    rb = torch.load(root_b / "nf-unsw-nb15-v2/models/multiclass/graphsage_cb_seed42/last.pt", weights_only=False)
    for k in ra["model"]:
        assert torch.allclose(ra["model"][k], rb["model"][k], atol=1e-6), k
    assert ra["optimizer"]["state"][0]["step"] == rb["optimizer"]["state"][0]["step"]


def test_bundle_reload_matches_and_model_learns(tmp_path):
    root = setup_root(tmp_path)
    state = train(args_for(root, epochs=8, seed=1))
    assert state["best_f1"] > 0.5  # du lieu tong hop tach duoc -> mo hinh phai hoc duoc, khong sup ve 1 lop
    run = root / "nf-unsw-nb15-v2/models/multiclass/graphsage_cb_seed1"
    m1, b = load_bundle(run / "best_bundle.pt")
    m2, _ = load_bundle(run / "best_bundle.pt")
    g = torch.load(shard_paths(root / "nf-unsw-nb15-v2/graphs/multiclass", "test")[0], weights_only=False)[0]
    with torch.no_grad():
        assert torch.equal(m1(g.x, g.edge_index, g.edge_attr), m2(g.x, g.edge_index, g.edge_attr))
    assert b["benign_class_id"] == 0 and b["model_config"]["num_classes"] == 3


def test_evaluate_test_writes_predictions_and_confusion(tmp_path):
    from models.train_gnn_v2 import evaluate_test

    root = setup_root(tmp_path)
    train(args_for(root, epochs=2))
    res = evaluate_test(args_for(root))
    run = root / "nf-unsw-nb15-v2/models/multiclass/graphsage_cb_seed42"
    pred = pd.read_parquet(run / "test_predictions.parquet")
    assert pred[FLOW_ID_COL].is_unique and len(pred) == res["flows"]
    cm = pd.read_csv(run / "test_confusion_raw.csv", index_col=0)
    assert cm.shape == (3, 3) and cm.to_numpy().sum() == len(pred)
    assert list(cm.index) == ["Benign", "DoS", "Scan"]  # thu tu lop co dinh theo mapping


def test_binary_derived_from_multiclass_graphs(tmp_path):
    from etl.protocol import build_label_mapping

    root = setup_root(tmp_path)
    ds = root / "nf-unsw-nb15-v2"
    # contract.json (nhu ETL ghi) co mapping cho ca 2 task; KHONG co thu muc graphs/binary
    attack = pd.Series(["Benign", "DoS", "Scan"])
    contract = {"label_mappings": {"binary": build_label_mapping(attack, "binary"),
                                   "multiclass": build_label_mapping(attack, "multiclass")}}
    (ds / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert not (ds / "graphs" / "binary").exists()

    args = parse_args(["--dataset", "nf-unsw-nb15-v2", "--task", "binary", "--processed", str(root),
                       "--hidden-dim", "8", "--batch-size", "4", "--epochs", "6", "--patience", "50"])
    state = train(args)
    assert state["best_f1"] > 0.5
    _, bundle = load_bundle(ds / "models" / "binary" / "graphsage_cb_seed42" / "best_bundle.pt")
    assert bundle["model_config"]["num_classes"] == 2 and bundle["benign_class_id"] == 0
    assert bundle["label_mapping"]["class_names"] == {0: "Benign", 1: "Attack"} or \
        {int(k): v for k, v in bundle["label_mapping"]["class_names"].items()} == {0: "Benign", 1: "Attack"}


def test_relabel_matches_directly_built_binary_labels():
    from models.train_gnn_v2 import load_shard
    import glob

    mc = sorted(glob.glob("data/processed_v2/nf-unsw-nb15-v2/graphs/multiclass/val_shard_*.pt"))
    bi = sorted(glob.glob("data/processed_v2/nf-unsw-nb15-v2/graphs/binary/val_shard_*.pt"))
    if not (mc and bi):
        import pytest
        pytest.skip("can du lieu UNSW local da dung do thi binary + multiclass")
    a, b = load_shard(__import__("pathlib").Path(mc[0]), benign_id=2), torch.load(bi[0], weights_only=False)
    assert len(a) == len(b)
    for ga, gb in zip(a[:20], b[:20]):
        assert torch.equal(ga.y, gb.y) and torch.equal(ga.flow_id, gb.flow_id)
