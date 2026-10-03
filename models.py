"""Model zoo and pipeline factory - shared by Tasks 1.2, 1.4, 1.5 and 1.6.

Kept separate from `task1_2_supervised.py` so that the later tasks can rebuild a
model without importing the whole tuning experiment.
"""
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import config
from task1_1_preprocessing import make_preprocessor

# Gradient boosting: LightGBM if installed, otherwise sklearn's histogram GBM,
# which uses the same algorithm family so the comparison still holds.
try:
    from lightgbm import LGBMClassifier

    def _gbm(**kw):
        return LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS,
                              verbose=-1, **kw)

    GBM_NAME = "LightGBM"
except ImportError:  # pragma: no cover - environment dependent
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _gbm(**kw):
        return HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, **kw)

    GBM_NAME = "HistGBM"

# The two boosters name the leaf-size parameter differently.
_GBM_GRID = {"clf__learning_rate": [0.03, 0.1, 0.2],
             "clf__max_depth": [3, 6, -1] if GBM_NAME == "LightGBM" else [3, 6, None]}
_GBM_GRID["clf__min_child_samples" if GBM_NAME == "LightGBM"
          else "clf__min_samples_leaf"] = [10, 30, 100]


# Model name -> (constructor, RandomizedSearchCV grid keyed on the "clf" step).
# Grids are deliberately small: the point is tuning *inside* CV, not a big sweep.
MODEL_SPECS = {
    "DecisionTree": (
        lambda **kw: DecisionTreeClassifier(random_state=config.RANDOM_STATE, **kw),
        {"clf__max_depth": [4, 6, 8, 12, None],
         "clf__min_samples_leaf": [1, 5, 20, 50],
         "clf__criterion": ["gini", "entropy"]},
    ),
    "kNN": (
        # Scale-sensitive: relies on Euclidean proximity, hence the scaler in the pipeline.
        lambda **kw: KNeighborsClassifier(n_jobs=config.N_JOBS, **kw),
        {"clf__n_neighbors": [3, 5, 11, 25],
         "clf__weights": ["uniform", "distance"],
         "clf__p": [1, 2]},
    ),
    "NaiveBayes": (
        lambda **kw: GaussianNB(**kw),
        {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    ),
    "LogReg": (
        lambda **kw: LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE, **kw),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        lambda **kw: RandomForestClassifier(random_state=config.RANDOM_STATE,
                                            n_jobs=config.N_JOBS, **kw),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [8, 16, None],
         "clf__min_samples_leaf": [1, 5, 20],
         "clf__max_features": ["sqrt", 0.5]},
    ),
    GBM_NAME: (_gbm, _GBM_GRID),
}

# Only these accept `class_weight`; used by the class-weight imbalance strategy.
SUPPORTS_CLASS_WEIGHT = {"DecisionTree", "LogReg", "RandomForest", "LightGBM", "HistGBM"}


def build_pipeline(model_name, sampler=None, **model_kwargs):
    """Preprocessor -> (optional resampler) -> classifier, as one estimator.

    Using imblearn's Pipeline is what makes the resampling leakage-free: the
    sampler is applied to the training part of each CV fold only, never to the
    held-out fold.
    """
    constructor, _ = MODEL_SPECS[model_name]
    steps = [("prep", make_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", constructor(**model_kwargs)))
    return Pipeline(steps)


def build_baseline():
    """Majority-class baseline: always predicts 'legit'."""
    return DummyClassifier(strategy="most_frequent")
