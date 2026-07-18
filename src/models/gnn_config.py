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

MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5  # dung neu val_f1_macro khong cai thien sau 5 epoch lien tiep

NODE_FEATURE_DIM = 4   # bac vao, bac ra, pagerank, clustering (graph/config.py NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = 39  # so dac trung luong mang da chuan hoa (giong feature_cols o ETL/Graph Builder)

DEFAULT_PROCESSED_DIR = Path("data/processed")
