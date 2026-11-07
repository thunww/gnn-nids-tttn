import numpy as np
import pandas as pd

from etl.protocol import build_label_mapping
from models.train_baseline_v2 import _full_proba, task_metrics


def test_task_metrics_handles_absent_class_without_crash():
    m = build_label_mapping(pd.Series(["Analysis", "Benign", "DoS"]), "multiclass")
    m["class_names"] = {int(k): v for k, v in m["class_names"].items()}
    y_true = np.array([1, 1, 2, 2])  # lop 0 (Analysis) vang o tap danh gia
    y_pred = np.array([1, 1, 2, 1])
    proba = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.6, 0.3]])
    r = task_metrics(y_true, y_pred, proba, m)
    assert r["per_class"]["Analysis"]["support"] == 0
    assert 0 < r["f1_macro"] < 1  # trung binh chi tren lop co mat


def test_full_proba_pads_classes_missing_in_train():
    class Fake:
        classes_ = np.array([0, 2])
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    p = _full_proba(Fake(), None, 3)
    assert p.shape == (1, 3) and p[0, 1] == 0 and p[0, 2] == 0.7
