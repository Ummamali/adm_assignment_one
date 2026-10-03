"""Task 1.6 - Explainability.

Global: impurity-based vs permutation importance for the best model.
Local: SHAP explanations for 3 true positives, 2 false positives, 1 false negative,
each with a two-line analyst-friendly summary.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

import config
import models
from task1_1_preprocessing import prepare_data
from task1_2_supervised import get_ranked_models
from utils import section, show_table, save_fig, score_of

SHAP_SAMPLE = 2000      # rows used for the global SHAP summary plot
PERM_SAMPLE = 5000      # permutation importance refits nothing but re-scores N times


def global_importance(name, pipe, data):
    """Impurity importance (from the fitted trees) vs permutation importance.

    They disagree for a reason: impurity importance is computed on the *training*
    data and is biased toward high-cardinality / continuous features that offer
    many split points, while permutation importance is measured on held-out data
    against the metric we actually care about (PR-AUC), and it splits credit
    between correlated features instead of giving it all to one.
    """
    clf = pipe.named_steps["clf"]
    features = list(pipe.named_steps["prep"].get_feature_names_out())

    if not hasattr(clf, "feature_importances_"):
        print(f"{name} exposes no impurity importance; reporting permutation only.")
        impurity = pd.Series(np.nan, index=features)
    else:
        impurity = pd.Series(clf.feature_importances_, index=features)

    X_s = data.X_test.sample(min(PERM_SAMPLE, len(data.X_test)),
                             random_state=config.RANDOM_STATE)
    perm = permutation_importance(pipe, X_s, data.y_test.loc[X_s.index],
                                  scoring="average_precision", n_repeats=5,
                                  random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)

    table = (pd.DataFrame({"Feature": features,
                           "Impurity": impurity.to_numpy(),
                           "Permutation": perm.importances_mean,
                           "PermStd": perm.importances_std})
             .sort_values("Permutation", ascending=False))
    show_table(table.head(15).round(5), f"Global feature importance - {name}",
               "16_global_importance.csv")

    top = table.head(12).set_index("Feature")
    fig, ax = plt.subplots(figsize=(8, 5))
    top[["Impurity", "Permutation"]].div(top[["Impurity", "Permutation"]].max()).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(xlabel="importance (each column scaled to its own max)",
           title=f"Impurity vs permutation importance - {name}")
    save_fig(fig, "16_global_importance.png")
    return table


def local_explanations(name, pipe, data, threshold=0.5):
    """SHAP for 3 TP, 2 FP and 1 FN, with a short written explanation for each."""
    try:
        import shap
    except ImportError:
        print("shap is not installed - run `pip install shap` to produce Task 1.6 locals.")
        return None

    clf = pipe.named_steps["clf"]
    X_t = pipe.named_steps["prep"].transform(data.X_test)
    scores = score_of(pipe, data.X_test)
    pred, truth = scores >= threshold, data.y_test.to_numpy().astype(bool)

    # Pick the cases: highest-confidence TPs/FPs, and the FN the model was most
    # wrong about (lowest score among the frauds it missed).
    cases = {"TP": np.where(pred & truth)[0][np.argsort(-scores[pred & truth])][:3],
             "FP": np.where(pred & ~truth)[0][np.argsort(-scores[pred & ~truth])][:2],
             "FN": np.where(~pred & truth)[0][np.argsort(scores[~pred & truth])][:1]}

    explainer, n_sample = _make_explainer(shap, clf, X_t)

    # Global SHAP summary on a sample, for the report.
    sample = X_t.sample(min(n_sample, len(X_t)), random_state=config.RANDOM_STATE)
    plt.figure()
    shap.summary_plot(_positive_class(explainer.shap_values(sample)), sample, show=False)
    save_fig(plt.gcf(), "16_shap_summary.png")

    rows = []
    for kind, idxs in cases.items():
        for i in idxs:
            row = X_t.iloc[[i]]
            sv = pd.Series(_positive_class(explainer.shap_values(row))[0], index=X_t.columns)
            top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(3)
            drivers = ", ".join(f"{f} ({'+' if v > 0 else '-'}{abs(v):.2f})"
                                for f, v in top.items())
            rows.append({"Case": kind, "Index": int(i), "Score": round(float(scores[i]), 4),
                         "Amount": round(float(data.X_test.iloc[i]["Amount"]), 2),
                         "TopDrivers": drivers})

    table = pd.DataFrame(rows)
    show_table(table, "Local SHAP explanations", "16_local_explanations.csv")

    print("\nANALYST SUMMARIES")
    for _, r in table.iterrows():
        verdict = {"TP": "Flagged, and it was fraud.",
                   "FP": "Flagged, but the customer confirmed it was genuine.",
                   "FN": "Not flagged, and it turned out to be fraud."}[r["Case"]]
        print(f"  [{r['Case']}] txn #{r['Index']}, {r['Amount']:.2f} units, "
              f"risk score {r['Score']:.3f}. {verdict}")
        print(f"      The score was driven mainly by {r['TopDrivers']} - these are "
              f"anonymised behavioural components, so read them as 'this transaction's "
              f"spending pattern sits far from this account's normal profile' "
              f"(+ pushes toward fraud, - pulls toward legitimate).")
    return table


def detector_agreement(data, anomaly_scores, importance_table, pipe, top_k=50):
    """Do the anomaly detector's top-ranked records look like fraud to the classifier?"""
    if not anomaly_scores:
        return None
    clf_scores = score_of(pipe, data.X_test)
    rows = []
    for det_name, s in anomaly_scores.items():
        det_top = set(np.argsort(s)[::-1][:top_k])
        clf_top = set(np.argsort(clf_scores)[::-1][:top_k])
        rows.append({"Detector": det_name,
                     f"OverlapTop{top_k}": len(det_top & clf_top) / top_k,
                     "FraudsInDetTop": int(data.y_test.to_numpy()[list(det_top)].sum()),
                     "FraudsInClfTop": int(data.y_test.to_numpy()[list(clf_top)].sum())})
    table = pd.DataFrame(rows).sort_values("FraudsInDetTop", ascending=False)
    show_table(table.round(3), f"Agreement between detectors and classifier (top {top_k})",
               "16_detector_agreement.csv")
    print("A low overlap means the two are looking at different things: the classifier "
          "ranks on learned fraud *patterns* (the features in the table above), the "
          "detector ranks on raw distance from the bulk of the data. The detector's extra "
          "alerts are where a genuinely new fraud pattern would first appear.")
    return table


def _make_explainer(shap, clf, X):
    """TreeExplainer when the model is a tree ensemble, KernelExplainer otherwise.

    KernelExplainer is model-agnostic but orders of magnitude slower, so it gets a
    much smaller background set and sample.
    """
    try:
        return shap.TreeExplainer(clf), SHAP_SAMPLE
    except Exception:
        background = shap.kmeans(X.sample(500, random_state=config.RANDOM_STATE), 10)
        print(f"  {type(clf).__name__} is not a tree model - falling back to the slower "
              f"model-agnostic KernelExplainer on a small sample.")
        return shap.KernelExplainer(lambda d: clf.predict_proba(d)[:, 1], background), 100


def _positive_class(shap_values):
    """Normalise shap_values across the shapes different boosters return."""
    arr = np.asarray(shap_values[-1] if isinstance(shap_values, list) else shap_values)
    if arr.ndim == 3:            # (rows, features, classes) -> positive class
        arr = arr[..., -1]
    return arr[None, :] if arr.ndim == 1 else arr   # KernelExplainer drops the row axis


def run(data, ranked, anomaly_scores=None):
    section("TASK 1.6 - Explainability")
    # The brief asks for impurity-based importance, which only tree models expose,
    # so explain the best tree model if the overall winner is not one.
    trees = [(n, p) for n, p in ranked
             if hasattr(p.named_steps["clf"], "feature_importances_")]
    name, pipe = (trees or ranked)[0]
    if name != ranked[0][0]:
        print(f"Best model ({ranked[0][0]}) has no impurity importance; "
              f"explaining the best tree-based model instead: {name}.")
    importance = global_importance(name, pipe, data)
    local = local_explanations(name, pipe, data)
    agreement = detector_agreement(data, anomaly_scores, importance, pipe)
    return {"global": importance, "local": local, "agreement": agreement}


if __name__ == "__main__":
    d = prepare_data()
    run(d, get_ranked_models(d, tune_first=False, top_k=1)[0])
