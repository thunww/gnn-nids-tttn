import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/train_gnn.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

from etl.config import DATASETS
from models.gat import GATEdgeClassifier
from models.gcn import GCNEdgeClassifier
from models.gnn_config import (
    BATCH_SIZE,
    DEFAULT_PROCESSED_DIR,
    DROPOUT,
    EDGE_FEATURE_DIM,
    GAT_HEADS,
    HIDDEN_DIM,
    LEARNING_RATE,
    NODE_FEATURE_DIM,
    NUM_EPOCHS,
    NUM_LAYERS,
)


def load_graphs(processed_dir: Path, folder_name: str, split: str) -> list:
    return torch.load(processed_dir / folder_name / f"{split}_graphs.pt", weights_only=False)


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


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_graphs: list,
    val_graphs: list,
    out_dir: Path,
    device: torch.device,
) -> None:
    loader_train = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = out_dir / "checkpoints" / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
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
        print(f"  epoch {epoch}/{NUM_EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1_macro={val_f1:.4f}")

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"{model_name}.pt")
    print(f"  luu model hoan chinh tai: {models_dir / f'{model_name}.pt'}")


def run(processed_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for folder_name in DATASETS:
        print(f"=== {folder_name} ===")
        train_graphs = load_graphs(processed_dir, folder_name, "train")
        val_graphs = load_graphs(processed_dir, folder_name, "val")
        num_classes = int(max(g.y.max().item() for g in train_graphs) + 1)
        print(f"  so do thi train={len(train_graphs)} val={len(val_graphs)} | so lop={num_classes}")

        out_dir = processed_dir / folder_name

        models = {
            "gcn": GCNEdgeClassifier(
                NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT
            ),
            "gat": GATEdgeClassifier(
                NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS, GAT_HEADS, DROPOUT
            ),
        }

        for name, model in models.items():
            print(f"--- {folder_name} / {name} ---")
            model = model.to(device)
            train_one_model(name, model, train_graphs, val_graphs, out_dir, device)


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
