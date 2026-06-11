import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted":  round(f1_score(y_true, y_pred, average="weighted"), 4),
        "f1_macro":     round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision":    round(precision_score(y_true, y_pred, average="weighted"), 4),
        "recall":       round(recall_score(y_true, y_pred, average="weighted"), 4),
    }
    if y_prob is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics

def print_report(name, y_true, y_pred):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

def confusion_matrix_df(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Pred Negative", "Pred Positive"],
    )

def results_table(all_metrics: dict):
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({"Model": model_name, **m})
    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("f1_weighted", ascending=False)
    return df
