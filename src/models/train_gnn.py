import csv
import json
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


def load_class_names(processed_dir: Path, folder_name: str, num_classes: int) -> list[str]:
    mapping_path = processed_dir / folder_name / "attack_label_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping.get(str(i), str(i)) for i in range(num_classes)]


def compute_class_weights(
    train_graphs: list, num_classes: int, device: torch.device, beta: float = CB_BETA
) -> torch.Tensor:
    """Class-Balanced Loss (Cui et al., CVPR 2019) -- trong so dua tren "so mau hieu qua"
    (1 - beta^n) / (1 - beta), bot cuc doan hon nhieu so voi ty le nghich truc tiep voi
    lop sieu hiem, giam bat on dinh khi train (da quan sat thay o CSE-CIC-IDS2018 luot truoc).
    """
    all_labels = torch.cat([g.y for g in train_graphs]).numpy()
    counts = np.bincount(all_labels, minlength=num_classes).astype(np.float64)
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
    train_graphs: list,
    val_graphs: list,
    out_dir: Path,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Path:
    loader_train = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        val_acc, val_f1 = evaluate(model, loader_val, device)
        avg_loss = total_loss / len(train_graphs)
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


def run(processed_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for folder_name in DATASETS:
        print(f"=== {folder_name} ===")
        train_graphs = load_graphs(processed_dir, folder_name, "train")
        val_graphs = load_graphs(processed_dir, folder_name, "val")
        num_classes = int(max(g.y.max().item() for g in train_graphs) + 1)
        print(f"  so do thi train={len(train_graphs)} val={len(val_graphs)} | so lop={num_classes}")

        class_weights = compute_class_weights(train_graphs, num_classes, device)
        class_names = load_class_names(processed_dir, folder_name, num_classes)

        out_dir = processed_dir / folder_name

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
            best_path = train_one_model(name, model, train_graphs, val_graphs, out_dir, device, class_weights)

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
    run(processed_dir)
