from pathlib import Path

from etl.config import ATTACK_COL, IDENTIFIER_COLS, LABEL_COL

SRC_IP_COL, SRC_PORT_COL, DST_IP_COL, DST_PORT_COL = IDENTIFIER_COLS

# So dong (flow) moi do thi con, va ty le chong lap giua 2 cua so lien tiep
WINDOW_SIZE = 10_000
WINDOW_OVERLAP = 0.5

NODE_FEATURE_NAMES = ["in_degree", "out_degree", "pagerank", "clustering"]

DEFAULT_PROCESSED_DIR = Path("data/processed")
