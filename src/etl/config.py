from pathlib import Path

IDENTIFIER_COLS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"]
LABEL_COL = "Label"
ATTACK_COL = "Attack"

# folder_name (trong data/raw/ và data/processed/) -> tên file csv thô
DATASETS = {
    "nf-cse-cic-ids2018-v2": "NF-CSE-CIC-IDS2018-v2.csv",
    "nf-unsw-nb15-v2": "NF-UNSW-NB15-v2.csv",
}

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
