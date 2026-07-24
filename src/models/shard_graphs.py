import sys
from pathlib import Path

# Cho phep chay truc tiep "python src/models/shard_graphs.py" tu bat ky thu muc nao.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from etl.config import DATASETS
from models.gnn_config import DEFAULT_PROCESSED_DIR

# So do thi/shard cho tap train -- muc tieu moi shard ~1.5-2GB de nap an toan tren RAM Colab
# free (~12-13GB). Chi chia tap train (duoc doc lai nhieu lan/epoch); tap val nho hon nhieu
# va chi doc 1 lan/epoch nen giu nguyen ca file. Xem docs/decisions.md ("chia shard du lieu do thi").
GRAPHS_PER_SHARD = 2_200


def shard_split(processed_dir: Path, folder_name: str, split: str, graphs_per_shard: int) -> None:
    src_path = processed_dir / folder_name / f"{split}_graphs.pt"
    if not src_path.exists():
        print(f"  [{folder_name}/{split}] khong tim thay {src_path}, bo qua")
        return

    graphs = torch.load(src_path, weights_only=False)
    if len(graphs) <= graphs_per_shard:
        print(f"  [{folder_name}/{split}] {len(graphs)} do thi <= {graphs_per_shard}, khong can chia shard")
        return

    num_shards = (len(graphs) + graphs_per_shard - 1) // graphs_per_shard
    print(f"  [{folder_name}/{split}] {len(graphs)} do thi -> {num_shards} shard")

    for i in range(num_shards):
        chunk = graphs[i * graphs_per_shard : (i + 1) * graphs_per_shard]
        out_path = processed_dir / folder_name / f"{split}_graphs_shard{i}.pt"
        torch.save(chunk, out_path)
        print(f"    da luu {out_path.name} ({len(chunk)} do thi)")


def run(processed_dir: Path, graphs_per_shard: int = GRAPHS_PER_SHARD) -> None:
    for folder_name in DATASETS:
        shard_split(processed_dir, folder_name, "train", graphs_per_shard)


if __name__ == "__main__":
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROCESSED_DIR
    run(processed_dir)
