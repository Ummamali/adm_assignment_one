"""Helpers shared by more than one task script: metrics, thresholds, I/O, plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # write files, never open a window
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve,
)

import config


# --- console ----------------------------------------------------------
def section(title):
    """Print a visible banner so the console log maps onto the report sections."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show_table(df, title=None, path=None):
    """Print a dataframe and optionally persist it as csv for the report."""
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False) if isinstance(df, pd.DataFrame) else df)
    if path:
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.TABLE_DIR / path, index=False)
    return df


def save_fig(fig, filename):
    """Save a figure under FIG_DIR and close it."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / filename, dpi=120)
    plt.close(fig)
    print(f"  saved figure -> figures/{filename}")


# --- metrics ----------------------------------------------------------
def score_of(model, X):
    """Continuous fraud score, whatever the estimator exposes.

    A model trained on a label budget so small that it saw no fraud has a single
    class, so predict_proba has one column; it scores everything as legitimate.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X))
    return model.decision_function(X)


def classification_metrics(y_true, y_score, threshold=0.5, label=""):
    """The rare-event metric set required by Task 1.3, as a flat dict."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": label,
        "Threshold": round(float(threshold), 4),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
    }


def best_f1_threshold(y_true, y_score):
    """Threshold-moving: the cut-off that maximises F1 (used on the validation set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    # precision_recall_curve returns one more point than thresholds
    return float(thresholds[np.argmax(f1[:-1])])


def precision_at_k(y_true, y_score, k=None):
    """Share of true positives among the k highest-scored records.

    k defaults to the number of real anomalies, which is the budget an analyst
    team would realistically review.
    """
    y_true = np.asarray(y_true)
    k = int(y_true.sum()) if k is None else k
    if k == 0:
        return np.nan
    top_k = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k].sum() / k)


def imbalance_ratio(y):
    """negatives : positives, the headline number for Task 1.1."""
    counts = pd.Series(y).value_counts()
    return counts.get(0, 0) / max(counts.get(1, 0), 1)
