import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/train_gnn_transfer.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch_geometric.loader import DataLoader

from models.gat import GATEdgeClassifier
from models.gcn import GCNEdgeClassifier
from models.gnn_config import (
    BATCH_SIZE,
    DEFAULT_PROCESSED_DIR,
    DROPOUT_GAT,
    DROPOUT_GCN,
    EDGE_FEATURE_DIM,
    FINE_TUNE_LEARNING_RATE,
    GAT_HEADS,
    HIDDEN_DIM,
    NODE_FEATURE_DIM,
    NUM_LAYERS,
    PRETRAINED_SOURCE,
)
from models.train_gnn import (
    compute_class_weights,
    compute_confusion,
    count_and_collect_labels,
    list_train_shards,
    load_class_names,
    load_graphs,
    load_transferable_weights,
    print_confusion_summary,
    save_confusion_matrix,
    train_one_model,
)


def run(processed_dir: Path, output_dir: Path | None = None) -> None:
    """Transfer learning: nap trong so da train tren bo NGUON (lon, da on dinh) lam diem
    khoi dau cho bo DICH (nho, thieu du lieu), roi fine-tune tiep voi learning rate thap hon.

    Day la THU NGHIEM BO SUNG cho RQ1 (cai thien ket qua within-dataset cua bo dich) --
    KHAC voi Thi nghiem 2 / RQ2 (train 1 bo, test thang sang bo kia, khong tinh chinh gi).
    Ket qua luu rieng (hau to "_transfer"), KHONG ghi de model train-tu-dau da co, de con
    doi chieu ca 2 cach trong bao cao.

    processed_dir/output_dir: xem docs-string cua models.train_gnn.run() -- cung quy uoc
    (processed_dir = doc du lieu, nen la ban sao dia cuc bo; output_dir = ghi checkpoint,
    nen la Drive). model NGUON (da pretrain) la SAN PHAM cua lan train truoc (ghi vao
    output_dir cua lan do), khong phai du lieu goc -- vi vay doc tu output_dir, khong phai
    processed_dir.
    """
    output_dir = output_dir or processed_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_classes = {
        "gcn": (GCNEdgeClassifier, dict(num_layers=NUM_LAYERS, dropout=DROPOUT_GCN)),
        "gat": (GATEdgeClassifier, dict(num_layers=NUM_LAYERS, heads=GAT_HEADS, dropout=DROPOUT_GAT)),
    }

    for target_folder, source_folder in PRETRAINED_SOURCE.items():
        print(f"=== Transfer learning: {source_folder} -> {target_folder} ===")

        train_shard_paths = list_train_shards(processed_dir, target_folder)
        val_graphs = load_graphs(processed_dir, target_folder, "val")
        num_train_graphs, all_train_labels = count_and_collect_labels(train_shard_paths)
        num_classes = int(all_train_labels.max().item()) + 1
        print(
            f"  so do thi train={num_train_graphs} (chia {len(train_shard_paths)} shard)"
            f" val={len(val_graphs)} | so lop={num_classes}"
        )

        class_weights = compute_class_weights(all_train_labels, num_classes, device)
        del all_train_labels
        class_names = load_class_names(processed_dir, target_folder, num_classes)

        out_dir = output_dir / target_folder
        source_models_dir = output_dir / source_folder / "models"

        for name, (cls, kwargs) in model_classes.items():
            print(f"--- {target_folder} / {name} (transfer tu {source_folder}) ---")

            source_path = source_models_dir / f"{name}_best.pt"
            if not source_path.exists():
                print(f"  BO QUA: chua co model nguon tai {source_path} -- can train {source_folder} truoc.")
                continue

            model = cls(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, num_classes, **kwargs).to(device)
            source_state = torch.load(source_path, map_location=device, weights_only=True)
            load_transferable_weights(model, source_state)

            best_path = train_one_model(
                f"{name}_transfer",
                model,
                train_shard_paths,
                num_train_graphs,
                val_graphs,
                out_dir,
                device,
                class_weights,
                learning_rate=FINE_TUNE_LEARNING_RATE,
            )

            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            loader_val = DataLoader(val_graphs, batch_size=BATCH_SIZE)
            cm = compute_confusion(model, loader_val, device, num_classes)

            cm_path = out_dir / "models" / f"{name}_transfer_confusion_matrix.csv"
            save_confusion_matrix(cm, class_names, cm_path)
            print_confusion_summary(cm, class_names)
            print(f"  confusion matrix luu tai: {cm_path}")


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    run(processed_dir, output_dir)
