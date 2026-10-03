"""Task 1.5 - Drift and unseen threats (Track A).

The model is refitted on the *time-ordered* training split - reusing the
random-split model here would be leakage, because it has already seen
transactions that occur after the time-ordered cut-off. The test period is then
cut into consecutive windows and precision/recall are tracked per window.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

import config
import models
from task1_1_preprocessing import prepare_data, make_preprocessor
from task1_2_supervised import get_ranked_models
from utils import section, show_table, save_fig, best_f1_threshold, score_of


def _windows(X, y, n):
    """Split the chronologically ordered test period into n consecutive windows."""
    order = X["Time"].to_numpy().argsort()
    for idx in np.array_split(order, n):
        yield X.iloc[idx], y.iloc[idx]


def run(data, model_name=None, n_windows=config.N_TIME_WINDOWS):
    section("TASK 1.5 - Drift over the test period")
    model_name = model_name or models.GBM_NAME
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = (data.time_splits[k]
                                                  for k in ("train", "val", "test"))
    print(f"Time-ordered split: train {X_tr.shape} ({y_tr.sum()} frauds), "
          f"val {X_val.shape} ({y_val.sum()}), test {X_te.shape} ({y_te.sum()})")

    # --- best supervised model, refitted on the past only ---------------
    sup = models.build_pipeline(model_name).fit(X_tr, y_tr)
    thr = best_f1_threshold(y_val, score_of(sup, X_val))   # threshold from the past, too
    print(f"Supervised threshold chosen on the time-ordered validation window: {thr:.3f}")

    # --- best anomaly detector, fitted on past normal traffic -----------
    pre = make_preprocessor().fit(X_tr)
    det = IsolationForest(n_estimators=200, random_state=config.RANDOM_STATE,
                          n_jobs=config.N_JOBS).fit(pre.transform(X_tr)[y_tr.to_numpy() == 0])
    # The detector has no probability, so it gets a fixed alert budget: flag the
    # top q% of each window, with q = the fraud rate observed in training.
    alert_rate = max(y_tr.mean(), 1e-4)

    rows = []
    for i, (Xw, yw) in enumerate(_windows(X_te, y_te, n_windows), start=1):
        sup_pred = (score_of(sup, Xw) >= thr).astype(int)
        anom_score = -det.score_samples(pre.transform(Xw))
        cutoff = np.quantile(anom_score, 1 - alert_rate)
        anom_pred = (anom_score >= cutoff).astype(int)

        for label, pred in [("Supervised", sup_pred), ("IsolationForest", anom_pred)]:
            rows.append({
                "Window": f"W{i}", "Model": label, "Rows": len(yw), "Frauds": int(yw.sum()),
                "Precision": precision_score(yw, pred, zero_division=0),
                "Recall": recall_score(yw, pred, zero_division=0),
            })

    table = pd.DataFrame(rows)
    show_table(table.round(4), f"Per-window performance ({n_windows} consecutive windows)",
               "15_drift_windows.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric in zip(axes, ["Recall", "Precision"]):
        for label, grp in table.groupby("Model"):
            ax.plot(grp["Window"], grp[metric], "o-", label=label)
        ax.set(title=metric, ylim=(0, 1.05))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Drift across the test period")
    save_fig(fig, "15_drift_windows.png")

    monitoring_plan()
    return table


def monitoring_plan():
    """Deployment proposal - no implementation required by the brief."""
    print("""
MONITORING AND RETRAINING PROPOSAL
  Track daily, with alerting thresholds:
    - Alert volume and precision-at-the-alert-budget, measured on the analyst
      dispositions that come back 24-72h later. A sustained drop of >20% relative to
      the trailing 4-week median triggers investigation.
    - Feature drift: population stability index per feature against the training
      window; PSI > 0.2 on any feature in the model's top 10 raises a warning.
    - Score drift: KS distance between today's score distribution and the training
      distribution. This catches drift before any label arrives, which matters
      because fraud labels are delayed by chargeback cycles.
  Retraining:
    - Scheduled monthly on a rolling 6-month window, so the model keeps pace with
      gradual behavioural change.
    - Event-driven whenever a drift alert fires or recall on confirmed frauds falls
      below the agreed SLA.
    - Every candidate is validated on a strictly later time window than it was
      trained on (the protocol used here), then shadow-deployed for one week before
      it takes over; the previous model stays warm for instant rollback.
  Covering the gap between retrains:
    - Run the unsupervised detector permanently alongside the classifier. It needs no
      labels, so it is the only component that can flag a genuinely new fraud pattern
      on day one; its top-ranked unexplained alerts feed the labelling queue and
      become next month's training data.
""")


if __name__ == "__main__":
    d = prepare_data()
    run(d, get_ranked_models(d, tune_first=False, top_k=1)[0][0][0])
