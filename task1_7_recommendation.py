"""Task 1.7 - Discussion and recommendation.

Task 1.7 is a written deliverable, so this module does not model anything. It
collects the headline numbers produced by Tasks 1.2-1.6 into one printed brief,
so every claim in the written page can be traced back to a result.
"""
from utils import section


def run(results):
    section("TASK 1.7 - Evidence for the recommendation")

    best_name, _ = results["supervised"]["ranked"][0]
    final = results["evaluation"]["metrics"]
    best_row = final.iloc[0]
    detectors = results["anomaly"]["detectors"]
    det_row = detectors.iloc[0]
    drift = results["drift"]

    print(f"Best supervised model          : {best_name}")
    print(f"  test PR-AUC / ROC-AUC        : {best_row['PR_AUC']:.4f} / {best_row['ROC_AUC']:.4f}")
    print(f"  precision / recall / MCC     : {best_row['Precision']:.3f} / "
          f"{best_row['Recall']:.3f} / {best_row['MCC']:.3f}")
    print(f"  cost-optimal threshold       : {results['evaluation']['cost_threshold']:.3f}")
    p = results["evaluation"]["mcnemar_p"]
    print(f"  McNemar vs 2nd best (p)      : {p:.4f}" if p is not None else "")

    print(f"Best anomaly detector          : {det_row['Detector']} ({det_row['Setting']})")
    print(f"  test PR-AUC / Precision@k    : {det_row['PR_AUC']:.4f} / {det_row['Precision@k']:.3f}")

    budget = results["anomaly"]["label_budget"]
    beats = budget.loc[budget["PR_AUC"] > detectors["PR_AUC"].max(), "Budget"]
    print(f"  supervision wins from        : "
          + (f"{beats.min():.0%} of labels" if len(beats) else "never, on this split"))

    sup_drift = drift[drift["Model"] == "Supervised"]
    print(f"Recall across the {sup_drift.shape[0]} time windows : "
          f"{[round(v, 3) for v in sup_drift['Recall']]}")

    print("""
WRITE-UP SKELETON (one page, cite the numbers above)
  1. Recommendation: a hybrid. The supervised model is the primary filter; the
     anomaly detector runs in parallel as the cover for patterns it has never seen.
  2. Why not supervised alone: it only detects fraud resembling labelled history,
     and the per-window recall above shows how it degrades as behaviour moves.
  3. Why not unsupervised alone: compare the two PR-AUC figures above - at equal
     alert budget the detector buries analysts in false alarms.
  4. Operating point: justify the cost-optimal threshold, not 0.5, using the test
     cost table from Task 1.3.
  5. Labelling policy: the label-budget curve says how much analyst labelling is
     worth buying before returns flatten.
  6. Trust: SHAP gives every alert a reason, which is what makes it actionable.
  7. Limitations: V1-V28 are PCA components and cannot be read directly; results
     come from two days of 2013 European card data; costs are assumed, not measured.
""")
