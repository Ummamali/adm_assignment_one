"""Task 1.2 - Supervised classification under extreme imbalance.

Two experiments:
  A) tune six classifiers + a majority baseline with stratified 5-fold CV on the
     training set only, optimising PR-AUC;
  B) for the two best models, compare five imbalance strategies.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold

import config
import models
from progress import step, track
from task1_1_preprocessing import prepare_data
from utils import section, show_table, classification_metrics, best_f1_threshold, score_of

CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
SCORING = "average_precision"   # = PR-AUC; accuracy is meaningless at a 0.17% prior


def tune(model_name, X_train, y_train):
    """Randomised search inside stratified 5-fold CV. Returns the refitted pipeline."""
    pipe = models.build_pipeline(model_name)
    _, grid = models.MODEL_SPECS[model_name]
    search = RandomizedSearchCV(
        pipe, grid, n_iter=config.N_SEARCH_ITER, scoring=SCORING, cv=CV,
        random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, refit=True)
    # A grid smaller than n_iter is searched exhaustively, i.e. fewer candidates.
    n_candidates = min(config.N_SEARCH_ITER, len(ParameterGrid(grid)))
    with step(f"CV search: {n_candidates} candidates x {config.CV_FOLDS} folds + refit",
              parallel_tasks=n_candidates * config.CV_FOLDS):
        search.fit(X_train, y_train)
    print(f"  {model_name:<13} CV PR-AUC={search.best_score_:.4f}  "
          f"params={ {k.replace('clf__', ''): v for k, v in search.best_params_.items()} }")
    return search.best_estimator_, search.best_score_


def tune_all(data):
    """Experiment A: baseline + six tuned classifiers, scored on the validation set."""
    print("\n[A] Hyperparameter tuning (5-fold CV on train only, scoring = PR-AUC)")

    # Majority-class baseline: establishes the floor every real model must beat.
    baseline = models.build_baseline().fit(data.X_train, data.y_train)
    rows = [classification_metrics(data.y_val, baseline.predict_proba(data.X_val)[:, 1],
                                   label="MajorityBaseline")]

    fitted = {}
    for name in track(models.MODEL_SPECS, "Tuning"):
        pipe, _ = tune(name, data.X_train, data.y_train)
        fitted[name] = pipe
        with step("Scoring the validation set"):
            rows.append(classification_metrics(data.y_val, score_of(pipe, data.X_val),
                                               label=name))

    table = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    show_table(table.round(4), "Validation results, default 0.5 threshold", "12_model_comparison.csv")
    return fitted, table


def compare_strategies(data, model_names):
    """Experiment B: five imbalance strategies for the two best models.

    (a) none, (b) class weights, (c) random undersampling, (d) SMOTE,
    (e) threshold moving - all five fitted on train, scored on validation.
    Resamplers live inside the pipeline, so they only ever see training rows.
    """
    print(f"\n[B] Imbalance strategies for: {', '.join(model_names)}")
    jobs = []
    for name in model_names:
        variants = {
            "a) none": (None, {}),
            "c) undersample": (RandomUnderSampler(random_state=config.RANDOM_STATE), {}),
            "d) SMOTE": (SMOTE(random_state=config.RANDOM_STATE), {}),
        }
        if name in models.SUPPORTS_CLASS_WEIGHT:
            variants["b) class weights"] = (None, {"class_weight": "balanced"})
        jobs += [(name, strategy, sampler, kwargs)
                 for strategy, (sampler, kwargs) in sorted(variants.items())]

    rows = []
    for name, strategy, sampler, kwargs in track(jobs, "Strategy fit",
                                                 describe=lambda j: f"{j[0]} / {j[1]}"):
        # One fit at a time here (no outer joblib parallelism), so the
        # estimator itself is free to use every core.
        pipe = models.build_pipeline(name, sampler=sampler, n_jobs=config.N_JOBS,
                                     **kwargs).fit(data.X_train, data.y_train)
        scores = score_of(pipe, data.X_val)
        rows.append({"Model": name, "Strategy": strategy,
                     **classification_metrics(data.y_val, scores, label=name)})

        # (e) threshold moving re-uses the untreated model, only the cut-off moves.
        if strategy == "a) none":
            thr = best_f1_threshold(data.y_val, scores)
            rows.append({"Model": name, "Strategy": "e) threshold moving",
                         **classification_metrics(data.y_val, scores, threshold=thr,
                                                  label=name)})

    table = pd.DataFrame(rows)[["Model", "Strategy", "Threshold", "TP", "FP", "FN",
                                "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"]]
    show_table(table.round(4), "Model x strategy x metrics (validation set)",
               "12_strategy_comparison.csv")
    print("\nWhy resampling must stay inside the CV folds: SMOTE interpolates between "
          "neighbouring frauds. Applied before splitting, synthetic points derived from a "
          "validation fraud end up in training, so the model is partly scored on rows it "
          "has already seen and PR-AUC is optimistically biased. Undersampling before "
          "splitting is just as bad in the other direction: it changes the class prior of "
          "the held-out fold, so precision no longer reflects the 0.17% reality.")
    return table


def get_ranked_models(data, tune_first=True, top_k=2):
    """Fitted pipelines ordered by validation PR-AUC. Downstream tasks start here.

    `tune_first=False` skips the search and uses library defaults - useful when a
    later task script is run on its own.
    """
    if tune_first:
        fitted, table = tune_all(data)
    else:
        fitted = {n: models.build_pipeline(n, n_jobs=config.N_JOBS).fit(data.X_train, data.y_train)
                  for n in track(models.MODEL_SPECS, "Fitting with default parameters")}
        table = pd.DataFrame([classification_metrics(data.y_val, score_of(p, data.X_val), label=n)
                              for n, p in fitted.items()]).sort_values("PR_AUC", ascending=False)
    order = [n for n in table["Model"] if n in fitted][:top_k]
    return [(n, fitted[n]) for n in order], table


def run(data, top_k=3):
    """Returns the top-k fitted pipelines (best first) and both result tables."""
    section("TASK 1.2 - Supervised classification under imbalance")
    ranked, table = get_ranked_models(data, tune_first=True, top_k=top_k)
    # The brief asks for the five strategies on the two best models only.
    strategy_table = compare_strategies(data, [n for n, _ in ranked[:2]])
    return {"ranked": ranked, "model_table": table, "strategy_table": strategy_table}


if __name__ == "__main__":
    run(prepare_data())
