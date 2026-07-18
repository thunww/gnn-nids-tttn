from pathlib import Path

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
GAT_HEADS = 4
BATCH_SIZE = 32

NODE_FEATURE_DIM = 4   # bac vao, bac ra, pagerank, clustering (graph/config.py NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = 39  # so dac trung luong mang da chuan hoa (giong feature_cols o ETL/Graph Builder)

DEFAULT_PROCESSED_DIR = Path("data/processed")
