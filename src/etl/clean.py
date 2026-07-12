import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean(df: pd.DataFrame, attack_col: str) -> tuple[pd.DataFrame, dict[int, str]]:
    df = df.drop_duplicates()
    df = df.dropna()

    encoder = LabelEncoder()
    df = df.copy()
    df["Attack_encoded"] = encoder.fit_transform(df[attack_col])
    mapping = {int(i): label for i, label in enumerate(encoder.classes_)}

    return df, mapping
