"""Phase 02: huan luyen E-GraphSAGE tren protocol Phase 01 (binary / multiclass).

Khac train_gnn.py (thuc tap):
- doc shard + meta.json cua Phase 01, so lop/mapping lay tu mapping da khoa (khong tu max(label));
- seed co dinh, reseed theo epoch => resume cho ket qua giong het chay lien tuc;
- last.pt luu model + optimizer + scheduler + epoch + best + RNG; best bundle luu du config de
  dung lai model (loader cho infer/XAI);
- validation / test tinh THEO FLOW DUY NHAT: flow nam trong nhieu cua so chong lap thi lay trung binh
  xac suat cac cua so;
- class weights tinh tren flow train duy nhat (khong dem lap do chong lap cua so).

    python src/models/train_gnn_v2.py --dataset nf-unsw-nb15-v2 --task multiclass --seed 42
    python src/models/train_gnn_v2.py ... --eval-test        # chi chay cuoi cung
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from etl.config import DATASETS
from etl.protocol import FLOW_ID_COL, TARGET_COLS
from models.gnn_config import (
    BATCH_SIZE,
    CB_BETA,
    DROPOUT_GCN,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_DIM,
    HIDDEN_DIM_BY_DATASET,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    NUM_LAYERS,
    WEIGHT_DECAY,
)
from models.graphsage import GraphSAGEEdgeClassifier
from models.train_baseline_v2 import task_metrics

LOSSES = ("ce", "cb", "weighted")


# --------------------------------------------------------------------------- tien ich


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shard_paths(graph_dir: Path, split: str) -> list[Path]:
    paths = sorted(graph_dir.glob(f"{split}_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"Khong co shard {split} trong {graph_dir} -- chay run_graph_builder_v2.py truoc")
    return paths


def load_meta(graph_dir: Path) -> dict:
    meta = json.loads((graph_dir / "meta.json").read_text(encoding="utf-8"))
    meta["label_mapping"]["class_names"] = {int(k): v for k, v in meta["label_mapping"]["class_names"].items()}
    return meta


def model_config(meta: dict, graph_dir: Path, hidden_dim: int, dropout: float, num_layers: int) -> dict:
    """Kich thuoc dau vao lay tu du lieu that (khong hard-code 43/39)."""
    g = torch.load(shard_paths(graph_dir, "val")[0], weights_only=False)[0]
    return {
        "node_in_dim": int(g.x.shape[1]),
        "edge_in_dim": int(g.edge_attr.shape[1]),
        "hidden_dim": hidden_dim,
        "num_classes": int(meta["label_mapping"]["num_classes"]),
        "num_layers": num_layers,
        "dropout": dropout,
    }


def build_model(cfg: dict) -> GraphSAGEEdgeClassifier:
    return GraphSAGEEdgeClassifier(
        cfg["node_in_dim"], cfg["edge_in_dim"], cfg["hidden_dim"], cfg["num_classes"], cfg["num_layers"], cfg["dropout"]
    )


def load_bundle(path: Path, device: torch.device | str = "cpu") -> tuple[GraphSAGEEdgeClassifier, dict]:
    """Nap model bundle (dung cho danh gia / infer / XAI): tra ve (model.eval(), bundle_meta)."""
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = build_model(bundle["model_config"])
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval(), bundle


# --------------------------------------------------------------------------- nhan / weights


def unique_train_flow_labels(paths: list[Path]) -> np.ndarray:
    """Nhan cua cac flow train DUY NHAT (mot flow nam trong 2 cua so chong lap chi tinh mot lan)."""
    ids, ys = [], []
    for p in paths:
        graphs = torch.load(p, weights_only=False)
        ids.append(torch.cat([g.flow_id for g in graphs]).numpy())
        ys.append(torch.cat([g.y for g in graphs]).numpy())
        del graphs
    ids, ys = np.concatenate(ids), np.concatenate(ys)
    _, first = np.unique(ids, return_index=True)
    gc.collect()
    return ys[first]


def class_weights(labels: np.ndarray, num_classes: int, loss: str, beta: float = CB_BETA) -> torch.Tensor | None:
    """None cho CE khong trong so. Lop vang train -> dem 1 (trong so huu han, khong chia 0)."""
    if loss == "ce":
        return None
    counts = np.clip(np.bincount(labels, minlength=num_classes).astype(np.float64), 1, None)
    if loss == "cb":  # Class-Balanced (Cui 2019)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:  # weighted: nghich ty le tan suat
        w = 1.0 / counts
    w = w / w.sum() * num_classes
    assert np.isfinite(w).all()
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- du doan theo flow


@torch.no_grad()
def predict_flows(model: nn.Module, paths: list[Path], device: torch.device, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Du doan tren cac shard, gop theo flow_id (trung binh xac suat qua cac cua so chua flow do).
    Tra ve DataFrame [flow_id, y_true, y_pred] + cot proba_<i>."""
    model.eval()
    ids, ys, probs = [], [], []
    for p in paths:
        graphs = torch.load(p, weights_only=False)
        for batch in DataLoader(graphs, batch_size=batch_size):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            probs.append(F.softmax(out, dim=1).cpu().numpy())
            ids.append(batch.flow_id.cpu().numpy())
            ys.append(batch.y.cpu().numpy())
        del graphs
    ids, ys, probs = np.concatenate(ids), np.concatenate(ys), np.concatenate(probs)

    order = np.argsort(ids, kind="stable")
    ids, ys, probs = ids[order], ys[order], probs[order]
    uniq, start, counts = np.unique(ids, return_index=True, return_counts=True)
    mean_p = np.add.reduceat(probs, start, axis=0) / counts[:, None]
    out = pd.DataFrame({FLOW_ID_COL: uniq, "y_true": ys[start], "y_pred": mean_p.argmax(axis=1)})
    out["windows"] = counts
    proba = pd.DataFrame(mean_p.astype("float32"), columns=[f"proba_{i}" for i in range(mean_p.shape[1])])
    return pd.concat([out, proba], axis=1)


def metrics_from_predictions(pred: pd.DataFrame, mapping: dict) -> dict:
    cols = [c for c in pred.columns if c.startswith("proba_")]
    return task_metrics(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(), pred[cols].to_numpy(), mapping)


def confusion(pred: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ma tran nham lan (raw va chuan hoa theo hang) theo thu tu lop co dinh cua mapping."""
    labels = sorted(mapping["class_names"])
    names = [mapping["class_names"][i] for i in labels]
    cm = np.zeros((len(labels), len(labels)), dtype="int64")
    np.add.at(cm, (pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()), 1)
    raw = pd.DataFrame(cm, index=names, columns=names)
    norm = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    return raw, norm


# --------------------------------------------------------------------------- train


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_dir = args.processed / args.dataset
    graph_dir = ds_dir / "graphs" / args.task
    meta = load_meta(graph_dir)
    mapping = meta["label_mapping"]

    hidden = args.hidden_dim or HIDDEN_DIM_BY_DATASET.get(args.dataset, HIDDEN_DIM)
    cfg = model_config(meta, graph_dir, hidden, DROPOUT_GCN, NUM_LAYERS)
    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_paths = shard_paths(graph_dir, "train")
    val_paths = shard_paths(graph_dir, "val")
    if args.max_train_shards:
        train_paths = train_paths[: args.max_train_shards]

    labels = unique_train_flow_labels(train_paths)
    weights = class_weights(labels, cfg["num_classes"], args.loss)
    absent = [mapping["class_names"][i] for i in range(cfg["num_classes"]) if (labels == i).sum() == 0]
    print(f"[{args.dataset}/{args.task}] device={device} flow train duy nhat={len(labels)} "
          f"lop vang train={absent or 'khong'} loss={args.loss} seed={args.seed}")

    set_seed(args.seed)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "no_improve": 0, "history": []}
    last_path = run_dir / "last.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state = ck["state"]
        print(f"  RESUME tu epoch {state['epoch']} (best_f1={state['best_f1']:.4f})")

    max_epochs = args.epochs or MAX_EPOCHS
    for epoch in range(state["epoch"] + 1, max_epochs + 1):
        set_seed(args.seed * 1000 + epoch)  # reseed theo epoch => resume cho ket qua giong het
        model.train()
        order = train_paths.copy()
        random.shuffle(order)
        total, n_graphs = 0.0, 0
        for p in order:
            graphs = torch.load(p, weights_only=False)
            for batch in DataLoader(graphs, batch_size=args.batch_size, shuffle=True):
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch.x, batch.edge_index, batch.edge_attr), batch.y)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch.num_graphs
                n_graphs += batch.num_graphs
            del graphs
        gc.collect()

        pred = predict_flows(model, val_paths, device, args.batch_size)
        val = metrics_from_predictions(pred, mapping)
        scheduler.step(val["f1_macro"])
        row = {"epoch": epoch, "loss": total / max(n_graphs, 1), "val_acc": val["accuracy"],
               "val_f1_macro": val["f1_macro"], "lr": optimizer.param_groups[0]["lr"]}
        state["history"].append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if val["f1_macro"] > state["best_f1"]:
            state.update(best_f1=val["f1_macro"], best_epoch=epoch, no_improve=0)
            torch.save(
                {"state_dict": model.state_dict(), "model_config": cfg, "task": args.task,
                 "label_mapping": mapping, "benign_class_id": mapping["benign_class_id"],
                 "feature_hash": meta["feature_hash"], "dataset": args.dataset, "seed": args.seed,
                 "loss": args.loss, "epoch": epoch, "val_f1_macro": val["f1_macro"]},
                run_dir / "best_bundle.pt",
            )
            (run_dir / "val_metrics.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
            raw, norm = confusion(pred, mapping)
            raw.to_csv(run_dir / "val_confusion_raw.csv")
            norm.to_csv(run_dir / "val_confusion_normalized.csv")
        else:
            state["no_improve"] += 1

        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, last_path)
        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)

        if state["no_improve"] >= (args.patience or EARLY_STOPPING_PATIENCE):
            print(f"  early stopping tai epoch {epoch}")
            break

    (run_dir / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "model_config": cfg, "torch": torch.__version__,
         "device": str(device), "absent_train_classes": absent, "train_shards_used": len(train_paths)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  best epoch={state['best_epoch']} val_f1_macro={state['best_f1']:.4f} -> {run_dir}")
    return state


# --------------------------------------------------------------------------- test


def evaluate_test(args: argparse.Namespace) -> dict:
    """Chay MOT LAN o cuoi: nap best bundle, du doan test theo flow, luu du doan + metric (toan test
    va chi 'flow moi' khong co vector trung train)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_dir = args.processed / args.dataset
    graph_dir = ds_dir / "graphs" / args.task
    run_dir = run_dir_for(args)
    model, bundle = load_bundle(run_dir / "best_bundle.pt", device)
    mapping = bundle["label_mapping"]
    mapping["class_names"] = {int(k): v for k, v in mapping["class_names"].items()}

    pred = predict_flows(model, shard_paths(graph_dir, "test"), device, args.batch_size)
    pred.to_parquet(run_dir / "test_predictions.parquet", index=False)
    res = {"full_test": metrics_from_predictions(pred, mapping), "flows": len(pred)}

    novel_path = ds_dir / "novel_flow_ids.parquet"
    if novel_path.exists():
        nv = pd.read_parquet(novel_path)
        keep = pred[FLOW_ID_COL].isin(nv.loc[nv["split"] == "test", FLOW_ID_COL])
        if keep.any():
            res["novel_only"] = {"flows": int(keep.sum()), **metrics_from_predictions(pred[keep], mapping)}
            res["novel_only"].pop("per_class", None)
    raw, norm = confusion(pred, mapping)
    raw.to_csv(run_dir / "test_confusion_raw.csv")
    norm.to_csv(run_dir / "test_confusion_normalized.csv")
    (run_dir / "test_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: round(v, 4) for k, v in res["full_test"].items() if isinstance(v, float)})
    return res


def run_dir_for(args: argparse.Namespace) -> Path:
    """Noi luu checkpoint/ket qua. Tren Colab tro --runs-dir vao Drive de resume duoc khi phien ngat."""
    base = args.runs_dir / args.dataset / args.task if args.runs_dir else args.processed / args.dataset / "models" / args.task
    return base / f"graphsage_{args.loss}_seed{args.seed}"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--task", choices=list(TARGET_COLS), required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v2"))
    ap.add_argument("--runs-dir", type=Path, default=None, help="noi luu checkpoint (vd thu muc tren Drive)")
    ap.add_argument("--loss", choices=LOSSES, default="cb")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--max-train-shards", type=int, default=None, help="chi de chay thu (ghi vao config.json)")
    ap.add_argument("--fresh", action="store_true", help="bo qua last.pt, train lai tu dau")
    ap.add_argument("--eval-test", action="store_true", help="danh gia TEST (chi chay cuoi cung)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    evaluate_test(a) if a.eval_test else train(a)
