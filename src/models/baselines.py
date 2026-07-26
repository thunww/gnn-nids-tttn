from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from models.config import RANDOM_STATE


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def build_xgboost(num_classes: int) -> XGBClassifier:
    # nhi phan (2026-07-26): "multi:softprob" + num_class=2 co loi tuong thich da biet cua
    # XGBoost -- .predict() tra ve mang 2 chieu (dang xac suat) thay vi nhan don, gay loi
    # "mix of binary and multilabel-indicator targets" khi tinh accuracy/f1. Dung dung
    # "binary:logistic" (khong truyen num_class) cho truong hop 2 lop.
    if num_classes == 2:
        return XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    return XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=num_classes,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
