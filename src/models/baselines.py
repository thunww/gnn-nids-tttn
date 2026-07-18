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
