from pathlib import Path

from etl.config import ATTACK_COL, IDENTIFIER_COLS, LABEL_COL

SRC_IP_COL, SRC_PORT_COL, DST_IP_COL, DST_PORT_COL = IDENTIFIER_COLS

# So dong (flow) moi do thi con -- co the khac nhau tung bo du lieu. CSE-CIC-IDS2018 dang
# on dinh o 2_000 (giam tu 10_000 ngay 2026-07-19, xem docs/decisions.md), giu nguyen.
# UNSW-NB15-v2 it host hon, nghi ngo can cua so lon hon 2_000 de du ngu canh hanh vi -- dang
# thu 5_000 (2026-07-19).
WINDOW_SIZE_BY_DATASET = {
    "nf-cse-cic-ids2018-v2": 2_000,
    "nf-unsw-nb15-v2": 5_000,
}
WINDOW_OVERLAP = 0.5

NODE_FEATURE_NAMES = ["in_degree", "out_degree", "pagerank", "clustering"]

DEFAULT_PROCESSED_DIR = Path("data/processed")
