from collections.abc import Iterator

import pandas as pd


def sliding_windows(df: pd.DataFrame, window_size: int, overlap: float = 0.5) -> Iterator[pd.DataFrame]:
    """Cat df (da xao/chia san tu Giai doan 1) thanh cac cua so lien tiep,
    moi cua so window_size dong, chong lap overlap ty le voi cua so truoc.
    """
    step = max(1, int(window_size * (1 - overlap)))
    n = len(df)

    for start in range(0, n, step):
        end = start + window_size
        if end > n:
            break
        yield df.iloc[start:end].reset_index(drop=True)
