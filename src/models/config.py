from pathlib import Path

from etl.config import ATTACK_COL, IDENTIFIER_COLS, LABEL_COL

TARGET_COL = "Attack_encoded"  # phan loai da lop (loai tan cong cu the), theo dung pham vi nghien cuu

NON_FEATURE_COLS = IDENTIFIER_COLS + [LABEL_COL, ATTACK_COL, TARGET_COL]

DEFAULT_PROCESSED_DIR = Path("data/processed")

RANDOM_STATE = 42
