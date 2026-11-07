"""Read-only Phase 00 audit; writes new evidence, never changes legacy data/models.

Example: python scripts/phase00_audit.py --scan-raw
Synthetic training/XAI probes establish compatibility, not detection performance.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def label_counts_csv(path: Path) -> dict:
    import pandas as pd

    counts = {name: Counter() for name in ("Label", "Attack")}
    rows = 0
    missing = Counter()
    for chunk in pd.read_csv(path, usecols=list(counts), chunksize=100_000):
        rows += len(chunk)
        for name in counts:
            missing[name] += int(chunk[name].isna().sum())
            counts[name].update({str(k): int(v) for k, v in chunk[name].value_counts().items()})
    return {"rows": rows, "counts": {k: dict(v) for k, v in counts.items()},
            "missing_labels": dict(missing), "scope": "full CSV, labels only; not a full data-quality scan"}


def parquet_summary(path: Path) -> dict:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    labels = [c for c in ("Label", "Attack", "Attack_encoded") if c in columns]
    counts = {c: Counter() for c in labels}
    for batch in parquet.iter_batches(batch_size=100_000, columns=labels):
        frame = batch.to_pandas()
        for col in labels:
            counts[col].update({str(k): int(v) for k, v in frame[col].value_counts(dropna=False).items()})
    return {"rows": parquet.metadata.num_rows, "columns": columns,
            "counts": {k: dict(v) for k, v in counts.items()}}


def inventory(data_root: Path) -> list[dict]:
    return [{"path": p.relative_to(data_root).as_posix(), "bytes": p.stat().st_size}
            for p in sorted(data_root.rglob("*")) if p.is_file()]


def graph_partition_projection(path: Path, window: int) -> dict:
    """Project the current builder's intervals without loading graph archives."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    step = window // 2
    count = max(0, (rows - window) // step + 1)
    train, val = int(count * 0.70), int(count * 0.15)
    slices = {"train": (0, train), "val": (train + 1, train + 1 + val),
              "test": (train + val + 2, count)}
    result = {}
    for name, (first, end) in slices.items():
        end = min(end, count)
        result[name] = {"graphs": max(0, end - first), "first_row": first * step,
                        "end_row_exclusive": (end - 1) * step + window if end > first else first * step,
                        "unique_flow_class_counts": Counter()}
    offset = 0
    for batch in parquet.iter_batches(batch_size=100_000, columns=["Attack"]):
        frame = batch.to_pandas()
        for part in result.values():
            lower = max(0, part["first_row"] - offset)
            upper = min(len(frame), part["end_row_exclusive"] - offset)
            if lower < upper:
                part["unique_flow_class_counts"].update({str(k): int(v) for k, v in
                    frame.iloc[lower:upper]["Attack"].value_counts().items()})
        offset += len(frame)
    for part in result.values():
        part["unique_flow_class_counts"] = dict(part["unique_flow_class_counts"])
    return {"scope": "projection from current code and source-order table; archives not deserialized",
            "window": window, "step": step, "total_graphs_before_purge": count, "splits": result}


def dataset_audit(data_root: Path, scan_raw: bool) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from api.zeek_convert import FEATURE_COLS
    from etl.config import DATASETS
    from graph.config import WINDOW_SIZE_BY_DATASET

    results = {}
    for folder, filename in DATASETS.items():
        raw = data_root / "raw" / folder / filename
        processed = data_root / "processed" / folder
        result = {"raw_exists": raw.exists(), "processed_exists": processed.exists()}
        if raw.exists():
            columns = list(pd.read_csv(raw, nrows=0).columns)
            result["raw_columns"] = columns
            result["missing_model_features"] = sorted(set(FEATURE_COLS) - set(columns))
            result["unexpected_columns"] = sorted(set(columns) - set(FEATURE_COLS) - {
                "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "Label", "Attack"})
            result["timestamp_columns"] = [c for c in columns if "TIME" in c.upper() or c.lower() in {"ts", "timestamp"}]
            # FLOW_DURATION is an interval, not an event timestamp.
            sample = pd.read_csv(raw, nrows=4096)
            numeric = sample.select_dtypes(include="number")
            result["raw_first_4096"] = {
                "rows": len(sample), "missing_cells": int(sample.isna().sum().sum()),
                "nonfinite_numeric_cells": int((~np.isfinite(numeric.to_numpy())).sum()),
                "duplicates": int(sample.duplicated().sum()),
                "scope": "prefix sample only; not full dataset validation"}
            result["raw_labels"] = label_counts_csv(raw) if scan_raw else {"status": "not_scanned"}
        result["parquet"] = {}
        for name in ("train", "val", "test", "full_chronological"):
            path = processed / f"{name}.parquet"
            if path.exists():
                result["parquet"][name] = parquet_summary(path)
        chronological = processed / "full_chronological.parquet"
        if chronological.exists():
            result["graph_partition_projection"] = graph_partition_projection(chronological, WINDOW_SIZE_BY_DATASET[folder])
        mapping = processed / "attack_label_mapping.json"
        if mapping.exists():
            result["mapping"] = json.loads(mapping.read_text(encoding="utf-8"))
            result["benign_multiclass_ids"] = [k for k, v in result["mapping"].items() if v.lower() == "benign"]
        scaler_path = processed / "scaler.joblib"
        upper_path = processed / "upper_bound.joblib"
        if scaler_path.exists() and upper_path.exists():
            scaler, upper = joblib.load(scaler_path), joblib.load(upper_path)
            names = list(scaler.feature_names_in_)
            result["scaler"] = {"n_features": scaler.n_features_in_,
                                "order_matches_converter": names == FEATURE_COLS,
                                "finite_mean_scale": bool(np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()),
                                "clip_order_matches": list(upper.index) == names,
                                "samples_seen": int(scaler.n_samples_seen_)}
        # Header/labels only: intentionally do not unpickle GB-sized RF or graph shards.
        result["heavy_artifacts_not_loaded"] = [p.name for p in processed.glob("*.pt")]
        results[folder] = result
        print(f"Dataset scanned: {folder}", flush=True)
    return results


def checkpoint_probe(data_root: Path) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from api.zeek_convert import FEATURE_COLS
    from etl.config import DATASETS
    from graph.build_graph import build_graph
    from models.graphsage import GraphSAGEEdgeClassifier
    from models.gnn_config import HIDDEN_DIM, HIDDEN_DIM_BY_DATASET

    results = {}
    for folder in DATASETS:
        directory = data_root / "processed" / folder
        path = directory / "models" / "graphsage_best.pt"
        table_path = directory / "train.parquet"
        if not path.exists() or not table_path.exists():
            results[folder] = {"status": "missing_checkpoint_or_table"}
            continue
        model = GraphSAGEEdgeClassifier(43, 39, HIDDEN_DIM_BY_DATASET.get(folder, HIDDEN_DIM), 2, 2)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        # Read a bounded batch, never load a multi-GB graph archive for this probe.
        frame = next(pq.ParquetFile(table_path).iter_batches(batch_size=128)).to_pandas()
        graph = build_graph(frame, FEATURE_COLS)
        model.eval()
        with torch.no_grad():
            output = model(graph.x, graph.edge_index, graph.edge_attr)
        assert output.shape == (len(frame), 2) and torch.isfinite(output).all()
        results[folder] = {"status": "passed", "graph_nodes": graph.num_nodes,
                           "edges": graph.num_edges, "output_shape": list(output.shape),
                           "binary_labels": sorted(np.unique(graph.y.numpy()).tolist()),
                           "scope": "strict weights reload + bounded forward only; no test evaluation"}
    return results


def synthetic_probe(output_dir: Path) -> dict:
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from models.graphsage import GraphSAGEEdgeClassifier
    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # A small artificial topology, explicitly unrelated to real attack labels.
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 0], [1, 2, 3, 4, 0, 0, 1, 3]], device=device)
    graphs = [Data(x=torch.randn(5, 43, device=device), edge_index=edge_index,
                   edge_attr=torch.randn(8, 39, device=device),
                   y=torch.arange(8, device=device) % 3) for _ in range(24)]
    model = GraphSAGEEdgeClassifier(43, 39, 32, 3, 2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_times = []
    losses = []
    for _ in range(2):
        started = time.perf_counter()
        model.train()
        total = 0.0
        for graph in graphs:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(graph.x, graph.edge_index, graph.edge_attr), graph.y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - started)
        losses.append(total / len(graphs))
    graph = graphs[0]
    model.eval()
    with torch.no_grad():
        before = model(graph.x, graph.edge_index, graph.edge_attr)
    saved_path = output_dir / "synthetic_3class_smoke.pt"
    torch.save(model.state_dict(), saved_path)
    reloaded = GraphSAGEEdgeClassifier(43, 39, 32, 3, 2, dropout=0.0).to(device).eval()
    reloaded.load_state_dict(torch.load(saved_path, map_location=device, weights_only=True))
    with torch.no_grad():
        after = reloaded(graph.x, graph.edge_index, graph.edge_attr)
    assert torch.allclose(before, after, atol=1e-7)
    explainer = Explainer(model=reloaded, algorithm=GNNExplainer(epochs=20),
                          explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
                          model_config={"mode": "multiclass_classification", "task_level": "edge", "return_type": "raw"})
    started = time.perf_counter()
    explanation = explainer(graph.x, graph.edge_index, edge_attr=graph.edge_attr, index=0)
    assert explanation.edge_mask.shape == (8,) and torch.isfinite(explanation.edge_mask).all()
    xai_seconds = time.perf_counter() - started
    ones, zeros = torch.ones(8, device=device), torch.zeros(8, device=device)
    with torch.no_grad():
        unmasked = explainer.get_masked_prediction(graph.x, graph.edge_index, edge_mask=ones, edge_attr=graph.edge_attr)
        no_messages = explainer.get_masked_prediction(graph.x, graph.edge_index, edge_mask=zeros, edge_attr=graph.edge_attr)
    message_delta = float((unmasked[0] - no_messages[0]).abs().max())
    assert message_delta > 1e-8, "Message mask has no measurable effect on this target"
    edge_attr = graph.edge_attr.detach().clone().requires_grad_(True)
    target_class = int(after[0].argmax())
    score = reloaded(graph.x, graph.edge_index, edge_attr)[0, target_class]
    gradient = torch.autograd.grad(score, edge_attr)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Controlled fixture proves why masking messages alone cannot explain the direct edge path.
    direct = GraphSAGEEdgeClassifier(43, 39, 4, 3, 2, dropout=0.0).to(device).eval()
    with torch.no_grad():
        for parameter in direct.parameters():
            parameter.zero_()
        direct.classifier[0].weight[0, 8] = 1.0
        direct.classifier[2].weight[1, 0] = 1.0
    direct_features = torch.zeros_like(graph.edge_attr)
    direct_features[0, 0] = 2.0
    assert float(direct(graph.x, graph.edge_index, direct_features)[0, 1].detach()) == 2.0
    direct_features[0, 0] = 0.0
    assert float(direct(graph.x, graph.edge_index, direct_features)[0, 1].detach()) == 0.0
    return {"status": "passed", "scope": "synthetic compatibility only; not multiclass NIDS performance",
            "device": str(device), "graphs": 24, "epochs": 2, "epoch_seconds": epoch_times,
            "loss": losses, "reload_max_abs_error": float((before-after).abs().max()),
            "checkpoint_bytes": saved_path.stat().st_size,
            "xai": {"algorithm": "GNNExplainer", "epochs": 20, "seconds": xai_seconds,
                    "edge_mask_shape": list(explanation.edge_mask.shape),
                    "node_mask_shape": list(explanation.node_mask.shape),
                    "target_edge_index": 0, "target_class": target_class,
                    "zero_vs_one_message_mask_logit_delta": message_delta,
                    "edge_attr_gradient_l1": float(gradient.abs().sum()),
                    "direct_edge_path_control_passed": True,
                    "limitation": "x fixed; no IG/Captum, no fidelity benchmark or trained-model explanation"},
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


def stream_probe(output_dir: Path) -> dict:
    import pandas as pd
    from api.zeek_convert import read_conn_log, convert_rows_to_dataframe, DEFAULT_MEAN_COLS, FEATURE_COLS
    from api.window_buffer import SlidingWindowBuffer
    from graph.windowing import sliding_windows

    fields = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
              "proto", "ip_proto", "duration", "orig_bytes", "resp_bytes", "orig_pkts",
              "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "history"]
    values = ["1", "fixture_0", "10.0.0.1", "1234", "10.0.0.2", "22", "tcp", "6",
              "0.5", "100", "200", "3", "4", "220", "360", "ShADadFf"]
    path = output_dir / "synthetic_conn.log"
    header = "#separator \\x09\n#fields\t" + "\t".join(fields) + "\n"
    path.write_text(header + "\t".join(values) + "\n", encoding="utf-8")
    rows = list(read_conn_log(path))
    frame = convert_rows_to_dataframe(rows, dict.fromkeys(DEFAULT_MEAN_COLS, 1.0))
    assert len(frame) == 1 and list(frame.columns)[4:] == FEATURE_COLS
    assert frame.iloc[0]["FLOW_DURATION_MILLISECONDS"] == 500.0
    buffer = SlidingWindowBuffer(4, 0.5)
    emitted = []
    for i in range(8):
        window = buffer.add({"id": i})
        if window is not None:
            emitted.append([r["id"] for r in window])
    offline = [w["id"].tolist() for w in sliding_windows(pd.DataFrame({"id": range(8)}), 4, 0.5)]
    assert emitted == offline == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]
    # Execute only the actual reader definition, avoiding unavailable FastAPI/model startup.
    tree = ast.parse((PROJECT / "src/api/realtime_server.py").read_text(encoding="utf-8"))
    reader_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tail_conn_log")
    code = "from __future__ import annotations\n" + ast.unparse(reader_node)
    namespace = {"Path": Path}
    exec(compile(code, "realtime_server_reader_probe", "exec"), namespace)
    reader = namespace["_tail_conn_log"](path)
    assert next(reader) is None  # Existing rows are skipped by the live reader.
    values[1] = "fixture_1"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\t".join(values) + "\n")
    assert next(reader)["uid"] == "fixture_1"
    reader.close()
    # Closing/recreating the source models truncate/replacement without OS rename restrictions.
    # The code itself has no reopen/rotation check; actual OS rotation stays a lab test.
    return {"status": "passed", "scope": "synthetic TSV + converter + exact reader definition; no live network",
            "converter_features": len(FEATURE_COLS), "buffer_matches_offline": True,
            "old_rows_skipped": True, "new_row_read": True,
            "rotation_live": "not tested; current reader has no reopen logic",
            "continuous_service": "not tested; FastAPI not installed locally"}


def real_pilot(data_root: Path, output_dir: Path) -> dict:
    """Bounded pilot on real source-order data; not a benchmark or final split."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from api.zeek_convert import FEATURE_COLS
    from graph.build_graph import build_graph
    from models.graphsage import GraphSAGEEdgeClassifier

    path = data_root / "processed/nf-cse-cic-ids2018-v2/full_chronological.parquet"
    if not path.exists():
        return {"status": "not_run", "reason": "real processed data not found"}
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frame = next(pq.ParquetFile(path).iter_batches(batch_size=24 * 64)).to_pandas()
    graphs = []
    for start in range(0, len(frame) - 63, 64):
        block = frame.iloc[start:start + 64]
        graph = build_graph(block, FEATURE_COLS)
        # Harness-only target substitution; production builder remains binary.
        graph.y = torch.tensor(block["Attack_encoded"].to_numpy(dtype="int64"))
        graphs.append(graph.to(device))
    mapping_path = path.parent / "attack_label_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    num_classes = len(mapping)
    model = GraphSAGEEdgeClassifier(43, 39, 128, num_classes, 2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    times, losses, rss = [], [], []
    for _ in range(2):
        model.train()
        started = time.perf_counter()
        total = 0.0
        for graph in graphs:
            optimizer.zero_grad()
            prediction = model(graph.x, graph.edge_index, graph.edge_attr)
            loss = torch.nn.functional.cross_entropy(prediction, graph.y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            try:
                import psutil
                rss.append(psutil.Process().memory_info().rss)
            except ImportError:
                pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        losses.append(total / len(graphs))
    graph = graphs[0]
    model.eval()
    with torch.no_grad():
        before = model(graph.x, graph.edge_index, graph.edge_attr)
    checkpoint = output_dir / "real_prefix_multiclass_pilot.pt"
    torch.save(model.state_dict(), checkpoint)
    clone = GraphSAGEEdgeClassifier(43, 39, 128, num_classes, 2, dropout=0.0).to(device).eval()
    clone.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    with torch.no_grad():
        after = clone(graph.x, graph.edge_index, graph.edge_attr)
    assert torch.isfinite(after).all() and torch.allclose(before, after, atol=1e-7)
    return {"status": "passed", "scope": "real prefix pilot; no held-out evaluation; not a deployable model",
            "source": str(path), "rows": len(frame), "graphs": len(graphs), "edges_per_graph": 64,
            "hidden_dim": 128, "num_output_classes": num_classes,
            "observed_class_ids": sorted(np.unique(frame["Attack_encoded"]).tolist()),
            "source_preprocessing": "legacy scaler; not leakage-corrected graduation pipeline",
            "epochs": 2, "device": str(device), "epoch_seconds": times, "loss": losses,
            "sampled_peak_process_rss_bytes": max(rss) if rss else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "reload_max_abs_error": float((before-after).abs().max()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "warning": "64-flow pilot differs from 2000-flow training windows; do not extrapolate full runtime"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scan-raw", action="store_true", help="Read full raw CSV labels in bounded chunks")
    parser.add_argument("--skip-dataset", action="store_true", help="Only run bounded technical probes")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir or PROJECT / "results/graduation/phase_00" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {"started_utc": stamp, "data_root": str(args.data_root.resolve()),
               "python": sys.version, "platform": platform.platform(), "sections": {}}
    sources = [Path(__file__)] + sorted((PROJECT / "src").rglob("*.py"))
    results["source_sha256"] = {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}
    packages = ("torch", "torch-geometric", "pandas", "scikit-learn", "pyarrow", "captum", "fastapi", "psutil")
    results["packages"] = {}
    for name in packages:
        try:
            results["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            results["packages"][name] = None
    results["tools"] = {name: shutil.which(name) for name in ("zeek", "zeekctl", "nvidia-smi")}
    try:
        import psutil
        results["memory"] = {"total_bytes": psutil.virtual_memory().total,
                             "available_start_bytes": psutil.virtual_memory().available}
    except ImportError:
        pass
    operations = [("inventory", lambda: inventory(args.data_root))]
    if not args.skip_dataset:
        operations += [("datasets", lambda: dataset_audit(args.data_root, args.scan_raw)),
                       ("legacy_checkpoints", lambda: checkpoint_probe(args.data_root))]
    operations += [("synthetic_train_xai", lambda: synthetic_probe(output_dir)),
                   ("synthetic_stream", lambda: stream_probe(output_dir)),
                   ("real_multiclass_pilot", lambda: real_pilot(args.data_root, output_dir))]
    failed = False
    for name, operation in operations:
        print(f"Running {name} ...", flush=True)
        started = time.perf_counter()
        try:
            value = operation()
            section = {"status": "passed", "seconds": time.perf_counter() - started, "result": value}
        except Exception as exc:
            section = {"status": "failed", "seconds": time.perf_counter() - started,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            failed = True
        results["sections"][name] = section
        try:
            import psutil
            results["rss_after_section_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            pass
        (output_dir / "audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {section['status']} ({section['seconds']:.2f}s)", flush=True)
    print(f"Evidence: {output_dir / 'audit.json'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
