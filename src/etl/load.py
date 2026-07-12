from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path, folder_name: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / folder_name / filename)
