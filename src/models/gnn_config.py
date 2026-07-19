from pathlib import Path

HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4  # L2 regularization, chong overfitting (dac biet cho GAT)
BATCH_SIZE = 32
GAT_HEADS = 4

# Dropout rieng cho tung kien truc -- GAT can dropout cao hon de chong overfitting
# tren he so attention (tham khao ReGrAt, arXiv:2211.14770)
DROPOUT_GCN = 0.4
DROPOUT_GAT = 0.5

MAX_EPOCHS = 80
# patience tang tu 5 -> 15 (2026-07-19): voi bo du lieu it batch/epoch (vd UNSW-NB15 chi ~10
# batch/epoch), val_f1_macro rat nhieu tung epoch -- patience ngan de kich hoat nham do nhieu
# ngau nhien, khoa cung mo hinh vao trang thai doan bua 1 lop qua som. Xem docs/decisions.md.
EARLY_STOPPING_PATIENCE = 15

# Class-Balanced Loss (Cui et al., CVPR 2019) -- he so beta cho cong thuc "so mau hieu qua"
CB_BETA = 0.999

# Tu giam learning rate khi val_f1_macro chung lai (ReduceLROnPlateau) -- GNN von noi tieng
# huan luyen khong on dinh, scheduler giup hoi tu muot hon. patience tang 3 -> 8 cung ly do tren.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8

NODE_FEATURE_DIM = 4   # bac vao, bac ra, pagerank, clustering (graph/config.py NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = 39  # so dac trung luong mang da chuan hoa (giong feature_cols o ETL/Graph Builder)

DEFAULT_PROCESSED_DIR = Path("data/processed")
