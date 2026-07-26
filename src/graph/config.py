from pathlib import Path

from etl.config import ATTACK_COL, IDENTIFIER_COLS, LABEL_COL

SRC_IP_COL, SRC_PORT_COL, DST_IP_COL, DST_PORT_COL = IDENTIFIER_COLS

# So dong (flow) moi do thi con -- co the khac nhau tung bo du lieu. CSE-CIC-IDS2018 dang
# on dinh o 2_000 (giam tu 10_000 ngay 2026-07-19, xem docs/decisions.md), giu nguyen.
# UNSW-NB15-v2: tung thu 5_000 (2026-07-19, gia thuyet "can cua so lon hon" -- KHONG cai
# thien nhu ky vong, xem luot 6 phase3_model_training.md). Giam manh xuong 500 (2026-07-24)
# theo huong khac: UNSW-NB15-v2 chi co 668 do thi train (qua it cho GNN hoc), cua so nho hon
# sinh nhieu do thi con hon tu cung luong flow tho (~10 lan) -- xem docs/decisions.md.
WINDOW_SIZE_BY_DATASET = {
    "nf-cse-cic-ids2018-v2": 2_000,
    "nf-unsw-nb15-v2": 500,
}
WINDOW_OVERLAP = 0.5

# 4 dac trung cau truc + 39 dac trung tong hop tu canh ke (trung binh cac canh vao/ra node do)
# -- xem graph/node_features.py va docs/decisions.md (2026-07-19, "lam giau dac trung node").
NODE_FEATURE_NAMES = ["in_degree", "out_degree", "pagerank", "clustering"]

DEFAULT_PROCESSED_DIR = Path("data/processed")
