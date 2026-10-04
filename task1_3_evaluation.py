"""Task 1.3 - Evaluating rare-event detectors correctly.

Full metric set on the test set, the accuracy paradox shown numerically, ROC/PR
curves for the top three models, a cost-optimal decision threshold, and a
McNemar significance test between the two best models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

import config
from progress import step, track
from task1_1_preprocessing import prepare_data
from task1_2_supervised import get_ranked_models
from utils import section, show_table, save_fig, classification_metrics, score_of


def accuracy_paradox(y_test):
    """Show numerically why 99.8% accuracy can mean a useless detector."""
    n, frauds = len(y_test), int(y_test.sum())
    acc = (n - frauds) / n
    print(f"A model that predicts 'legit' for all {n} test transactions scores "
          f"{acc:.4%} accuracy while catching 0 of {frauds} frauds.")
    print(f"Its recall, precision, F1 and MCC are all 0, and the {frauds} frauds it misses "
          f"are the only rows the bank actually loses money on. Accuracy is dominated by "
          f"the {100 * acc:.2f}% majority class, so it cannot distinguish this model from a "
          f"perfect one; PR-AUC and MCC can.")


def final_metrics(ranked, data):
    """Confusion matrix + precision/recall/F1/MCC/ROC-AUC/PR-AUC for every final model."""
    rows = [classification_metrics(data.y_test, score_of(p, data.X_test), label=n)
            for n, p in track(ranked, "Test-set metrics")]
    table = pd.DataFrame(rows)
    return show_table(table.round(4), "Final test-set metrics", "13_final_metrics.csv")


def plot_curves(ranked, data, top_k=3):
    """ROC and PR curves for the top models on shared axes."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, pipe in track(ranked[:top_k], "ROC/PR curve"):
        s = score_of(pipe, data.X_test)
        fpr, tpr, _ = roc_curve(data.y_test, s)
        prec, rec, _ = precision_recall_curve(data.y_test, s)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(data.y_test, s):.3f})")
        ax_pr.plot(rec, prec, label=f"{name} (AP={average_precision_score(data.y_test, s):.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    baseline = data.y_test.mean()
    ax_pr.axhline(baseline, ls="--", c="k", lw=0.8, label=f"chance ({baseline:.4f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    save_fig(fig, "13_roc_pr_curves.png")

    print("\nWhy ROC looks optimistic here: the false positive rate has ~57,000 legitimate "
          "transactions in its denominator, so 500 false alarms move it by less than 0.01 "
          "and the curve stays hugging the top-left corner. Precision uses those same 500 "
          "false alarms against roughly 100 frauds, so it collapses. The PR curve therefore "
          "reflects the analyst's actual workload; ROC does not.")


def cost_analysis(name, pipe, data):
    """Cost matrix: a missed fraud costs its Amount, a false alarm a fixed fee.

    The threshold is chosen on validation and only then applied to test.
    """
    def total_cost(y_true, scores, amounts, threshold):
        pred = scores >= threshold
        missed = (y_true == 1) & ~pred            # false negatives: lose the full amount
        false_alarm = (y_true == 0) & pred        # false positives: investigation cost
        return amounts[missed].sum() + false_alarm.sum() * config.COST_FALSE_ALARM

    val_s, test_s = score_of(pipe, data.X_val), score_of(pipe, data.X_test)
    val_amt = data.X_val["Amount"].to_numpy()
    test_amt = data.X_test["Amount"].to_numpy()
    y_val, y_test = data.y_val.to_numpy(), data.y_test.to_numpy()

    grid = np.linspace(0.001, 0.999, 300)
    costs = [total_cost(y_val, val_s, val_amt, t) for t in grid]
    best_t = float(grid[int(np.argmin(costs))])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(best_t, c="r", ls="--", label=f"cost-optimal t={best_t:.3f}")
    ax.set(xlabel="decision threshold", ylabel="total cost on validation",
           title=f"Cost curve - {name}")
    ax.legend()
    save_fig(fig, "13_cost_curve.png")

    rows = [{"Threshold": f"default 0.500", "TestCost": total_cost(y_test, test_s, test_amt, 0.5)},
            {"Threshold": f"cost-optimal {best_t:.3f}",
             "TestCost": total_cost(y_test, test_s, test_amt, best_t)}]
    table = pd.DataFrame(rows)
    show_table(table.round(2),
               f"Cost on test set ({name}; FN = Amount, FP = {config.COST_FALSE_ALARM:.0f} units)",
               "13_cost.csv")
    return best_t


def mcnemar(ranked, data, threshold=0.5):
    """Exact McNemar test on the test set between the two best models.

    Compares only the rows where the two models disagree; an exact binomial test
    on the discordant pairs avoids the chi-square approximation, which is unsafe
    when disagreements are rare.
    """
    (n1, p1), (n2, p2) = ranked[0], ranked[1]
    c1 = (score_of(p1, data.X_test) >= threshold) == data.y_test.to_numpy()
    c2 = (score_of(p2, data.X_test) >= threshold) == data.y_test.to_numpy()
    b = int((c1 & ~c2).sum())   # only model 1 is right
    c = int((~c1 & c2).sum())   # only model 2 is right
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"\nMcNemar ({n1} vs {n2}): {n1}-only correct = {b}, {n2}-only correct = {c}, "
          f"p = {p:.4f}")
    print("  -> difference is " + ("statistically significant at alpha=0.05."
                                   if p < 0.05 else "NOT significant at alpha=0.05."))
    return p


def run(data, ranked):
    section("TASK 1.3 - Evaluating rare-event detectors correctly")
    accuracy_paradox(data.y_test)
    metrics = final_metrics(ranked, data)
    plot_curves(ranked, data)
    with step(f"Cost-optimal threshold for {ranked[0][0]}"):
        best_t = cost_analysis(ranked[0][0], ranked[0][1], data)
    p = None
    if len(ranked) > 1:
        with step("McNemar test between the two best models"):
            p = mcnemar(ranked, data)
    return {"metrics": metrics, "cost_threshold": best_t, "mcnemar_p": p}


if __name__ == "__main__":
    d = prepare_data()
    run(d, get_ranked_models(d, tune_first=False, top_k=3)[0])
