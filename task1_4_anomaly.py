"""Task 1.4 - Unsupervised and semi-supervised anomaly detection.

Three detectors (Isolation Forest, Local Outlier Factor, One-Class SVM), each in
two settings, scored with labels used *only* for evaluation. Then a label-budget
experiment that asks when supervision starts to pay for itself.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import OneClassSVM

import config
import models
from task1_1_preprocessing import prepare_data, make_preprocessor
from task1_2_supervised import get_ranked_models
from utils import section, show_table, save_fig, precision_at_k, score_of


def make_detectors():
    """Fresh detector instances. `novelty=True` lets LOF score unseen rows."""
    return {
        "IsolationForest": IsolationForest(n_estimators=200, contamination="auto",
                                           random_state=config.RANDOM_STATE,
                                           n_jobs=config.N_JOBS),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=config.N_JOBS),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.01),
    }


def _subsample(X, n, seed=config.RANDOM_STATE):
    """LOF and One-Class SVM scale quadratically, so they are fitted on a subsample."""
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed)


def evaluate_detectors(data):
    """Each detector x {unsupervised, semi-supervised}, scored on the test set."""
    pre = make_preprocessor().fit(data.X_train)          # fitted on train only
    X_tr, X_te = pre.transform(data.X_train), pre.transform(data.X_test)
    X_normal = X_tr[data.y_train.to_numpy() == 0]        # labels used for the fit *design* only

    settings = {
        # (i) fit on training data that still contains the frauds
        "unsupervised": X_tr,
        # (ii) fit on normal records only - the semi-supervised / novelty setting
        "semi-supervised": X_normal,
    }

    rows, scores = [], {}
    for setting, X_fit in settings.items():
        for name, det in make_detectors().items():
            fit_data = X_fit if name == "IsolationForest" else _subsample(
                X_fit, config.ANOMALY_SUBSAMPLE)
            det.fit(fit_data)
            # score_samples is "higher = more normal"; negate so higher = more anomalous.
            s = -det.score_samples(X_te)
            scores[f"{name} ({setting})"] = s
            rows.append({
                "Detector": name, "Setting": setting, "FitRows": len(fit_data),
                "ROC_AUC": roc_auc_score(data.y_test, s),
                "PR_AUC": average_precision_score(data.y_test, s),
                "Precision@k": precision_at_k(data.y_test, s),
            })
            print(f"  {name:<16} {setting:<16} PR-AUC={rows[-1]['PR_AUC']:.4f}")

    table = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    show_table(table.round(4),
               f"Anomaly detectors on the test set (k = {int(data.y_test.sum())} true frauds)",
               "14_anomaly_detectors.csv")
    return table, scores


def label_budget_curve(data, model_name, detector_table):
    """Retrain the best supervised model on 1/5/10/25/100% of the training labels.

    Only the labelled subset is available to the model, so this simulates the real
    constraint: analyst time to label, not data volume.
    """
    print("\nLabel-budget experiment")
    rows = []
    for budget in config.LABEL_BUDGETS:
        if budget < 1.0:
            X_b, _, y_b, _ = train_test_split(
                data.X_train, data.y_train, train_size=budget,
                stratify=data.y_train, random_state=config.RANDOM_STATE)
        else:
            X_b, y_b = data.X_train, data.y_train
        pipe = models.build_pipeline(model_name).fit(X_b, y_b)
        pr_auc = average_precision_score(data.y_test, score_of(pipe, data.X_test))
        rows.append({"Budget": budget, "LabelledRows": len(X_b),
                     "Frauds": int(y_b.sum()), "PR_AUC": pr_auc})
        print(f"  {budget:>6.0%} labels ({int(y_b.sum())} frauds) -> PR-AUC {pr_auc:.4f}")

    table = pd.DataFrame(rows)
    show_table(table.round(4), "PR-AUC vs label budget", "14_label_budget.csv")

    best_unsup = detector_table["PR_AUC"].max()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["Budget"] * 100, table["PR_AUC"], "o-", label=f"{model_name} (supervised)")
    for _, r in detector_table.sort_values("PR_AUC", ascending=False).head(3).iterrows():
        ax.axhline(r["PR_AUC"], ls="--", lw=1,
                   label=f"{r['Detector']} ({r['Setting']})")
    ax.set(xscale="log", xlabel="% of training labels used (log scale)",
           ylabel="test PR-AUC", title="Label budget vs unsupervised baselines")
    ax.legend(fontsize=8)
    save_fig(fig, "14_label_budget.png")

    crossover = table.loc[table["PR_AUC"] > best_unsup, "Budget"]
    print(f"\nSupervision overtakes the best anomaly detector (PR-AUC {best_unsup:.4f}) at a "
          + (f"label budget of {crossover.min():.0%}." if len(crossover)
             else "budget beyond 100% - i.e. not at all on this split."))
    return table


def run(data, model_name=None):
    section("TASK 1.4 - Unsupervised and semi-supervised anomaly detection")
    table, scores = evaluate_detectors(data)
    model_name = model_name or models.GBM_NAME
    budget = label_budget_curve(data, model_name, table)
    return {"detectors": table, "scores": scores, "label_budget": budget}


if __name__ == "__main__":
    d = prepare_data()
    run(d, get_ranked_models(d, tune_first=False, top_k=1)[0][0][0])
