import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame, stratify_col: str, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia 70% train / 15% val / 15% test, giữ tỷ lệ nhãn theo stratify_col."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df[stratify_col], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[stratify_col], random_state=seed
    )
    return train, val, test
