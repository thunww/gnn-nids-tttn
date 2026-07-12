from pathlib import Path

import pandas as pd


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Ep kieu du lieu ve dang nho hon de giam RAM: int64->so nho nhat vua du,
    float64->float32, cot it gia tri lap lai nhieu (Attack) -> category."""
    int_cols = df.select_dtypes(include="int64").columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

    if "Attack" in df.columns:
        df["Attack"] = df["Attack"].astype("category")

    return df


def load_raw(raw_dir: Path, folder_name: str, filename: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / folder_name / filename)
    return _downcast(df)
