from pathlib import Path

from etl.config import ATTACK_COL, ATTACK_ENCODED_COL, IDENTIFIER_COLS, LABEL_COL

# nhi phan (2026-07-26, doi tu da lop): xem docs/decisions.md -- chi con GraphSAGE, uu tien
# du doan "co tan cong hay khong" thay vi loai tan cong cu the.
TARGET_COL = LABEL_COL

NON_FEATURE_COLS = IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, ATTACK_ENCODED_COL, TARGET_COL]

DEFAULT_PROCESSED_DIR = Path("data/processed")

RANDOM_STATE = 42
