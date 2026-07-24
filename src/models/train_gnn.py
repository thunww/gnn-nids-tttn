import csv
import gc
import json
import random
import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/train_gnn.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

from etl.config import DATASETS
from models.gat import GATEdgeClassifier
from models.gcn import GCNEdgeClassifier
from models.gnn_config import (
    BATCH_SIZE,
    CB_BETA,
    DEFAULT_PROCESSED_DIR,
    DROPOUT_GAT,
    DROPOUT_GCN,
    EARLY_STOPPING_PATIENCE,
    EDGE_FEATURE_DIM,
    GAT_HEADS,
    HIDDEN_DIM,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    NODE_FEATURE_DIM,
    NUM_LAYERS,
    WEIGHT_DECAY,
)


def load_graphs(processed_dir: Path, folder_name: str, split: str) -> list:
    return torch.load(processed_dir / folder_name / f"{split}_graphs.pt", weights_only=False)


def list_train_shards(processed_dir: Path, folder_name: str) -> list[Path]:
    """Tra ve danh sach file shard cua tap train (train_graphs_shard0.pt, shard1.pt, ...),
    da sap xep. Neu chua chia shard (vd bo du lieu nho nhu UNSW-NB15, xem shard_graphs.py)
    thi tra ve [train_graphs.pt] goc (1 phan tu) -- tuong thich nguoc, khong can code rieng.
    """
    base_dir = processed_dir / folder_name
    shards = sorted(
        base_dir.glob("train_graphs_shard*.pt"),
        key=lambda p: int(p.stem.rsplit("shard", 1)[1]),
    )
    if shards:
        return shards
    return [base_dir / "train_graphs.pt"]


def count_and_collect_labels(shard_paths: list[Path]) -> tuple[int, torch.Tensor]:
    """Duyet 1 luot qua cac shard train de dem tong so do thi va gom nhan (y) -- dung de tinh
    so lop va class weights ma KHONG can giu ca danh sach Data trong RAM cung luc (moi luc chi
    1 shard, giai phong ngay sau khi lay xong nhan). Day la buoc chinh giup train tren Colab
    free (~12-13GB RAM) khong bi OOM voi bo dac trung node 43 chieu -- xem docs/decisions.md.
    """
    total = 0
    label_chunks = []
    for path in shard_paths:
        graphs = torch.load(path, weights_only=False)
        total += len(graphs)
        label_chunks.append(torch.cat([g.y for g in graphs]))
        del graphs
    gc.collect()
    return total, torch.cat(label_chunks)


def load_class_names(processed_dir: Path, folder_name: str, num_classes: int) -> list[str]:
    mapping_path = processed_dir / folder_name / "attack_label_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping.get(str(i), str(i)) for i in range(num_classes)]


def load_transferable_weights(model: nn.Module, source_state_dict: dict) -> None:
    """Nap trong so tu model nguon (vd da train tren bo du lieu khac), CHI nap cac tham so
    co dung kich thuoc -- bo qua (giu nguyen khoi tao ngau nhien) lop nao lech kich thuoc,
    dien hinh la lop phan loai cuoi cung neu 2 bo du lieu co so luop khac nhau (15 vs 10 lop).
    """
    model_state = model.state_dict()
    transferable = {
        k: v for k, v in source_state_dict.items() if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in source_state_dict if k not in transferable]
    model_state.update(transferable)
    model.load_state_dict(model_state)
    print(f"  da nap {len(transferable)}/{len(source_state_dict)} tham so tu model nguon (bo qua: {skipped})")


def compute_class_weights(
    all_labels: torch.Tensor, num_classes: int, device: torch.device, beta: float = CB_BETA
) -> torch.Tensor:
    """Class-Balanced Loss (Cui et al., CVPR 2019) -- trong so dua tren "so mau hieu qua"
    (1 - beta^n) / (1 - beta), bot cuc doan hon nhieu so voi ty le nghich truc tiep voi
    lop sieu hiem, giam bat on dinh khi train (da quan sat thay o CSE-CIC-IDS2018 luot truoc).
    """
    counts = np.bincount(all_labels.numpy(), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes  # chuan hoa: trung binh trong so ~ 1

    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return acc, f1_macro


def compute_confusion(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
    # labels=range(num_classes) ep du kich thuoc NxN, tranh truong hop 1 lop nao do
    # khong xuat hien trong tap danh gia lam sklearn tu bo lop do khoi ma tran.
    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + row.tolist())


def print_confusion_summary(cm: np.ndarray, class_names: list[str]) -> None:
    print("  Confusion matrix - lop hay bi nham thanh gi nhat (tren val):")
    for i, name in enumerate(class_names):
        row = cm[i].copy()
        total = int(row.sum())
        if total == 0:
            continue
        correct = int(row[i])
        row[i] = -1  # loai duong cheo de tim lop hay bi nham nhat
        worst_idx = int(row.argmax())
        if row[worst_idx] <= 0:
            print(f"    {name}: dung {correct}/{total} ({correct / total:.1%})")
            continue
        worst_name = class_names[worst_idx]
        worst_count = int(row[worst_idx])
        print(
            f"    {name}: dung {correct}/{total} ({correct / total:.1%})"
            f" | hay bi nham thanh '{worst_name}' ({worst_count} lan)"
        )


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_shard_paths: list[Path],
    num_train_graphs: int,
    val_graphs: list,
    out_dir: Path,
    device: torch.device,
    class_weights: torch.Tensor,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    loader_val = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_dir = out_dir / "checkpoints" / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{model_name}_best.pt"

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        # Nap tung shard vao RAM, train het batch cua shard do roi giai phong ngay, thay vi
        # giu toan bo train_graphs trong RAM suot qua trinh train (xem shard_graphs.py,
        # docs/decisions.md). Xao thu tu shard moi epoch de gan voi shuffle toan cuc hon.
        shard_order = train_shard_paths.copy()
        random.shuffle(shard_order)
        for shard_path in shard_order:
            shard_graphs = torch.load(shard_path, weights_only=False)
            loader_train = DataLoader(shard_graphs, batch_size=BATCH_SIZE, shuffle=True)
            for batch in loader_train:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            del shard_graphs, loader_train
        gc.collect()

        val_acc, val_f1 = evaluate(model, loader_val, device)
        avg_loss = total_loss / num_train_graphs
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch}/{MAX_EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1_macro={val_f1:.4f}  lr={lr:.6f}"
        )

        scheduler.step(val_f1)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
            print(f"  dung som (early stopping) tai epoch {epoch}")
            break

    print(f"  model tot nhat: epoch {best_epoch}  val_f1_macro={best_val_f1:.4f}  luu tai {best_path}")
    return best_path


def run(processed_dir: Path, output_dir: Path | None = None) -> None:
    """processed_dir: noi doc du lieu do thi (*_graphs*.pt) -- tren Colab nen tro vao ban sao
    o dia cuc bo (/content/...), vi shard train duoc doc lai moi epoch, doc truc tiep tu Drive
    (FUSE mount, thong luong thap) se rat cham. output_dir: noi ghi checkpoint/model (mac dinh
    = processed_dir neu khong truyen) -- tren Colab nen tro thang vao Drive de khong mat tien do
    neu phien bi ngat giua chung (file checkpoint nho, ghi truc tiep len Drive khong cham).
    """
    output_dir = output_dir or processed_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for folder_name in DATASETS:
        print(f"=== {folder_name} ===")
        train_shard_paths = list_train_shards(processed_dir, folder_name)
        val_graphs = load_graphs(processed_dir, folder_name, "val")
        num_train_graphs, all_train_labels = count_and_collect_labels(train_shard_paths)
        num_classes = int(all_train_labels.max().item()) + 1
        print(
            f"  so do thi train={num_train_graphs} (chia {len(train_shard_paths)} shard)"
            f" val={len(val_graphs)} | so lop={num_classes}"
        )

        class_weights = compute_class_weights(all_train_labels, num_classes, device)
        del all_train_labels
        class_names = load_class_names(processed_dir, folder_name, num_classes)

        out_dir = output_dir / folder_name

        models = {
            "gcn": GCNEdgeClassifier(
                NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT_GCN
            ),
            "gat": GATEdgeClassifier(
                NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS, GAT_HEADS, DROPOUT_GAT
            ),
        }

        for name, model in models.items():
            print(f"--- {folder_name} / {name} ---")
            model = model.to(device)
            best_path = train_one_model(
                name, model, train_shard_paths, num_train_graphs, val_graphs, out_dir, device, class_weights
            )

            # Nap lai dung trong so tot nhat (khong phai epoch cuoi) truoc khi tinh confusion matrix
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            loader_val = DataLoader(val_graphs, batch_size=BATCH_SIZE)
            cm = compute_confusion(model, loader_val, device, num_classes)

            cm_path = out_dir / "models" / f"{name}_confusion_matrix.csv"
            save_confusion_matrix(cm, class_names, cm_path)
            print_confusion_summary(cm, class_names)
            print(f"  confusion matrix day du luu tai: {cm_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    run(processed_dir, output_dir)
