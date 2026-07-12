import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scale(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    clip_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Clip outlier (theo ngưỡng percentile của train) rồi fit StandardScaler trên train,
    áp dụng lại (không fit) cho val/test. Dữ liệu NetFlow có cột kiểu bytes/giây có thể
    ra giá trị phi lý (chia cho thời lượng flow gần 0) nên cần clip trước khi tính phương sai,
    tránh tràn số học.
    """
    scaler = StandardScaler()
    train = train.copy()
    val = val.copy()
    test = test.copy()

    upper_bound = train[feature_cols].quantile(clip_quantile)
    train[feature_cols] = train[feature_cols].clip(upper=upper_bound, axis=1)
    val[feature_cols] = val[feature_cols].clip(upper=upper_bound, axis=1)
    test[feature_cols] = test[feature_cols].clip(upper=upper_bound, axis=1)

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    return train, val, test, scaler
