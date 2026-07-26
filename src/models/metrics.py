from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_full_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Tinh du 6 chi so danh gia theo docs/00_research_plan.md muc 6.1: Accuracy (tham khao),
    Precision (macro), Recall (macro), F1-macro (chinh), AUC-ROC, MCC.

    y_proba: xac suat du doan cho lop duong (nhan=1), mang 1 chieu -- can de tinh AUC-ROC.
    Bai toan hien tai la nhi phan (xem docs/decisions.md 2026-07-26); ham nay CHUA ho tro
    AUC-ROC da lop (can doi sang "one-vs-rest" neu sau nay quay lai da lop).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    metrics["auc_roc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    return metrics
