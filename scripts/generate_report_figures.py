"""Sinh cac hinh anh minh hoa cho bao cao, tu du lieu that cua project.
Chay: python scripts/generate_report_figures.py (tu thu muc goc D:\\GNN-NIDS_TTTN)
Luu ket qua vao report_figures/, PNG, >=200 DPI.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 220
plt.rcParams["savefig.dpi"] = 220

OUT_DIR = PROJECT_ROOT / "report_figures"
OUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Mau co dinh cho 3 model, dung nhat quan xuyen suot tat ca hinh (bang mau colorblind-safe
# cua seaborn) -- xem tuan thu 00_tong_quan.md / skill dataviz "categorical hue co dinh".
MODEL_COLORS = {
    "random_forest": "#4C72B0",
    "xgboost": "#DD8452",
    "graphsage": "#55A868",
}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "graphsage": "E-GraphSAGE",
}
DATASET_LABELS = {
    "nf-cse-cic-ids2018-v2": "NF-CSE-CIC-IDS2018-v2",
    "nf-unsw-nb15-v2": "NF-UNSW-NB15-v2",
}


def fig1_f1_comparison():
    df = pd.read_csv(PROCESSED_DIR / "test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, dataset in zip(axes, ["nf-cse-cic-ids2018-v2", "nf-unsw-nb15-v2"]):
        sub = df[df["dataset"] == dataset]
        models = ["random_forest", "xgboost", "graphsage"]
        values = [sub[sub["model"] == m]["f1_macro"].iloc[0] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        labels = [MODEL_LABELS[m] for m in models]

        bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.003,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_ylim(0.90, 1.00)
        ax.set_ylabel("F1-macro")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("So sánh F1-macro — Thí nghiệm 1 (tập test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = OUT_DIR / "tn1_f1_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_graphsage_predictions(folder_name: str):
    """Nap lai model GraphSAGE da train, chay suy luan tren tap test, tra ve (y_true, y_pred)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from models.gnn_config import BATCH_SIZE, EDGE_FEATURE_DIM, HIDDEN_DIM, HIDDEN_DIM_BY_DATASET, NODE_FEATURE_DIM, NUM_LAYERS
    from models.graphsage import GraphSAGEEdgeClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_graphs = torch.load(PROCESSED_DIR / folder_name / "test_graphs.pt", weights_only=False)

    hidden_dim = HIDDEN_DIM_BY_DATASET.get(folder_name, HIDDEN_DIM)
    model = GraphSAGEEdgeClassifier(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, hidden_dim, 2, NUM_LAYERS)
    model.load_state_dict(
        torch.load(PROCESSED_DIR / folder_name / "models" / "graphsage_best.pt", map_location=device, weights_only=True)
    )
    model = model.to(device).eval()

    loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
    return np.array(all_labels), np.array(all_preds)


def fig2_confusion_matrices():
    out_paths = []
    class_names = ["Benign", "Attack"]

    for folder_name, out_name in [
        ("nf-cse-cic-ids2018-v2", "confusion_matrix_cse_cic.png"),
        ("nf-unsw-nb15-v2", "confusion_matrix_unsw.png"),
    ]:
        print(f"  dang suy luan GraphSAGE tren tap test cua {folder_name}...")
        y_true, y_pred = _load_graphsage_predictions(folder_name)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        annot = np.array(
            [[f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Số lượng mẫu"},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            annot_kws={"fontsize": 11},
        )
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"Ma trận nhầm lẫn — E-GraphSAGE\n{DATASET_LABELS[folder_name]} (tập test)", fontweight="bold")
        fig.tight_layout()
        out_path = OUT_DIR / out_name
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def fig3_tn2_heatmap():
    # Du lieu that, da co san trong docs/graphsage/03_ket_qua.md (Thi nghiem 2)
    rows = ["CSE-CIC → UNSW-NB15", "UNSW-NB15 → CSE-CIC"]
    cols = ["Random Forest", "XGBoost", "E-GraphSAGE"]

    f1_data = np.array(
        [
            [0.4899, 0.5065, 0.4698],
            [0.4465, 0.4701, 0.3502],
        ]
    )
    mcc_data = np.array(
        [
            [0.0000, 0.1080, -0.0577],
            [-0.1056, -0.0329, -0.2430],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    sns.heatmap(
        f1_data,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=cols,
        yticklabels=rows,
        cbar_kws={"label": "F1-macro"},
        linewidths=0.8,
        linecolor="white",
        ax=axes[0],
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    axes[0].set_title("F1-macro", fontweight="bold")

    sns.heatmap(
        mcc_data,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=cols,
        yticklabels=rows,
        cbar_kws={"label": "MCC"},
        linewidths=0.8,
        linecolor="white",
        ax=axes[1],
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title("MCC (Matthews Correlation Coefficient)", fontweight="bold")

    fig.suptitle("Thí nghiệm 2 — Đánh giá chéo bộ dữ liệu (cross-dataset)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = OUT_DIR / "tn2_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig4_zscore_distribution():
    import joblib

    from etl.config import ATTACK_COL, ATTACK_ENCODED_COL, IDENTIFIER_COLS, LABEL_COL
    from etl.scale import apply_scale

    target, source = "nf-unsw-nb15-v2", "nf-cse-cic-ids2018-v2"
    print(f"  dang tinh lai z-score cho {target} theo thang do {source}...")

    target_test = pd.read_parquet(PROCESSED_DIR / target / "test.parquet")
    exclude = IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, ATTACK_ENCODED_COL]
    feature_cols = [c for c in target_test.columns if c not in exclude]

    target_scaler = joblib.load(PROCESSED_DIR / target / "scaler.joblib")
    raw_like = target_test.copy()
    raw_like[feature_cols] = target_scaler.inverse_transform(target_test[feature_cols])

    source_scaler = joblib.load(PROCESSED_DIR / source / "scaler.joblib")
    source_upper_bound = joblib.load(PROCESSED_DIR / source / "upper_bound.joblib")
    rescaled = apply_scale(raw_like, feature_cols, source_scaler, source_upper_bound)

    z_values = rescaled[feature_cols].to_numpy().flatten()
    z_abs = np.abs(z_values)
    z_abs_clipped = np.clip(z_abs, 0, 20)  # gioi han truc X de hinh de doc, khong mat thong tin ty le

    pct_extreme = (z_abs > 5).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 20, 81)
    counts, bin_edges, patches = ax.hist(z_abs_clipped, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)

    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 5:
            patch.set_facecolor("#C44E52")

    ax.axvline(5, color="black", linestyle="--", linewidth=1.5)
    ax.text(5.3, ax.get_ylim()[1] * 0.92, "Ngưỡng |z| = 5", fontsize=10, fontweight="bold")

    ax.text(
        0.97,
        0.85,
        f"{pct_extreme:.1f}% giá trị\nnằm ở vùng cực đoan\n(|z-score| > 5)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#C44E52",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#C44E52"),
    )

    normal_patch = mpatches.Patch(color="#4C72B0", label="Trong phạm vi bình thường (|z| ≤ 5)")
    extreme_patch = mpatches.Patch(color="#C44E52", label="Vùng cực đoan (|z| > 5)")
    ax.legend(handles=[normal_patch, extreme_patch], loc="upper right", bbox_to_anchor=(0.97, 0.72))

    ax.set_xlabel("|z-score| (dữ liệu UNSW-NB15 sau khi quy đổi sang thang đo CSE-CIC-IDS2018)")
    ax.set_ylabel("Số lượng giá trị đặc trưng")
    ax.set_title("Phân phối |z-score| — minh hoạ nguyên nhân sụp đổ ở Thí nghiệm 2", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / "zscore_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _box(ax, x, y, w, h, text, facecolor="#EAF2FB", edgecolor="#4C72B0", fontsize=10.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#333333"
    )
    ax.add_patch(arrow)


def fig5_pipeline_overview():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    steps = [
        "CSV thô\n(NetFlow V2)",
        "ETL\n(clean/split/scale)",
        "full_chronological\n.parquet",
        "Graph Builder\n(sliding window)",
        "*_graphs.pt",
        "E-GraphSAGE\n(train trên Colab)",
        "Đánh giá\n(test + cross-dataset)",
    ]
    n = len(steps)
    box_w, box_h = 1.55, 1.3
    gap = (13 - n * box_w) / (n + 1)
    y = (3.6 - box_h) / 2

    xs = []
    x = gap
    for text in steps:
        _box(ax, x, y, box_w, box_h, text, fontsize=9.5)
        xs.append(x)
        x += box_w + gap

    for i in range(n - 1):
        x1 = xs[i] + box_w
        x2 = xs[i + 1]
        _arrow(ax, x1, y + box_h / 2, x2, y + box_h / 2)

    ax.set_title("Quy trình xử lý dữ liệu và huấn luyện — tổng quan", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    out_path = OUT_DIR / "pipeline_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig6_egraphsage_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_w, box_h = 2.5, 1.6
    box4_w = 2.6
    gap = 0.35
    y_top = 3.6
    xs = [0.4]
    for _ in range(2):
        xs.append(xs[-1] + box_w + gap)
    xs.append(xs[-1] + box_w + gap)  # vi tri box 4, chieu rong khac box_w nen tinh rieng

    _box(
        ax,
        xs[0],
        y_top,
        box_w,
        box_h,
        "Bước 1 — Message\n\nφ(x_v, e_vu) = W₁·[x_v ; e_vu]\n\n(kết hợp đặc trưng hàng xóm\nv + đặc trưng cạnh v→u)",
        facecolor="#EAF2FB",
        edgecolor="#4C72B0",
        fontsize=9.5,
    )
    _box(
        ax,
        xs[1],
        y_top,
        box_w,
        box_h,
        "Bước 2 — Aggregate\n\na = Σ_{v∈N(u)} φ_v\n\n(tổng hợp thông điệp\ntừ tất cả hàng xóm)",
        facecolor="#FBEFEA",
        edgecolor="#DD8452",
        fontsize=9.5,
    )
    _box(
        ax,
        xs[2],
        y_top,
        box_w,
        box_h,
        "Bước 3 — Update\n\nh_u = σ(W₂·[x_u ; a])\n\n(cập nhật embedding\ncủa đỉnh u)",
        facecolor="#EAF6EC",
        edgecolor="#55A868",
        fontsize=9.5,
    )
    _box(
        ax,
        xs[3],
        y_top,
        box4_w,
        box_h,
        "Bước 4 — Phân loại\n\n[h_u; h_v; e_uv]\n→ MLP\n→ Benign/Attack",
        facecolor="#F5EAF6",
        edgecolor="#8172B2",
        fontsize=9.5,
    )

    for i in range(3):
        _arrow(ax, xs[i] + box_w, y_top + box_h / 2, xs[i + 1], y_top + box_h / 2)

    ax.text(
        6,
        1.6,
        "Lặp lại bước 1-3 qua K lớp truyền thông điệp (K = NUM_LAYERS = 2) trước khi sang bước 4.\n"
        "x_v, x_u: embedding node  |  e_vu, e_uv: đặc trưng cạnh gốc  |  σ: hàm kích hoạt (ReLU)  |  W₁, W₂: ma trận trọng số học được",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    ax.set_title(
        "Kiến trúc E-GraphSAGE — 1 lớp truyền thông điệp (Lo et al., 2021)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "egraphsage_architecture.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    created = []
    print("=== Ảnh 1: So sánh F1-macro (TN1) ===")
    created.append(fig1_f1_comparison())
    print("=== Ảnh 2: Confusion matrix (GraphSAGE) ===")
    created.extend(fig2_confusion_matrices())
    print("=== Ảnh 3: Heatmap TN2 (cross-dataset) ===")
    created.append(fig3_tn2_heatmap())
    print("=== Ảnh 4: Phân phối z-score ===")
    created.append(fig4_zscore_distribution())
    print("=== Ảnh 5: Sơ đồ pipeline ===")
    created.append(fig5_pipeline_overview())
    print("=== Ảnh 6: Sơ đồ kiến trúc E-GraphSAGE ===")
    created.append(fig6_egraphsage_architecture())

    print("\nĐã tạo xong các file:")
    for p in created:
        print(" -", p)
