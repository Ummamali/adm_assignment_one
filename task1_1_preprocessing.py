"""Task 1.1 - Data understanding, cleaning, split protocols and leakage audit.

Everything downstream imports `prepare_data()` from here, so the cleaning rules
and the splits are defined in exactly one place.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import config
from utils import section, save_fig, imbalance_ratio

RAW_FEATURES = ["Time", "Amount"]   # the only non-PCA columns; the ones that need scaling
TARGET = "Class"


@dataclass
class Dataset:
    """Both evaluation protocols, produced once and shared by every later task."""
    # Protocol (i): stratified random 70/15/15
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    # Protocol (ii): time-ordered 70/15/15, keyed by "train" / "val" / "test"
    time_splits: dict = field(default_factory=dict)
    features: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Profiling
# ----------------------------------------------------------------------
def profile(df):
    """Shape, dtypes, missing values, duplicates and class balance."""
    print(f"Shape: {df.shape}")
    print(f"Dtypes: {df.dtypes.value_counts().to_dict()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Infinite values: {int(np.isinf(df.select_dtypes('number')).sum().sum())}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    counts = df[TARGET].value_counts().sort_index()
    print(f"\nClass counts: {counts.to_dict()}")
    print(f"Fraud rate: {100 * counts.get(1, 0) / len(df):.3f}%")
    print(f"Imbalance ratio (legit:fraud) = {imbalance_ratio(df[TARGET]):.0f}:1")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Legit (0)", "Fraud (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log")            # without a log scale the fraud bar is invisible
    ax.set_ylabel("count (log scale)")
    ax.set_title("Class distribution")
    save_fig(fig, "11_class_distribution.png")


def plot_top_features(df, n=10):
    """Distributions of the n features most correlated with the target."""
    corr = df.drop(columns=[TARGET]).corrwith(df[TARGET]).abs().sort_values(ascending=False)
    top = corr.head(n).index.tolist()
    print(f"\nTop {n} informative features (|corr| with Class): {corr.head(n).round(3).to_dict()}")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, col in zip(axes.ravel(), top):
        for cls, color in [(0, "#4C78A8"), (1, "#E45756")]:
            ax.hist(df.loc[df[TARGET] == cls, col], bins=50, density=True,
                    alpha=0.55, color=color, label=f"Class {cls}")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Distribution of the 10 most informative features, by class")
    save_fig(fig, "11_top_feature_distributions.png")
    return top


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def clean(df):
    """Drop duplicates, non-finite rows, constant features and collinear pairs.

    Every removal is justified in the printed log so it can be quoted in the report.
    """
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n0 - len(df)} duplicate rows "
          f"(a duplicate split across train and test is a memorised answer).")

    # Infinite values: none in Track A, but the guard keeps the pipeline honest.
    finite = np.isfinite(df.select_dtypes("number")).all(axis=1)
    if (~finite).any():
        print(f"Dropped {int((~finite).sum())} rows containing non-finite values.")
        df = df[finite].reset_index(drop=True)

    constant = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    if constant:
        print(f"Dropped constant features (zero variance -> zero information): {constant}")
        df = df.drop(columns=constant)

    # Highly correlated pairs: keep the first feature of each pair, drop the redundant one.
    feats = df.drop(columns=[TARGET, "Time"])
    corr = feats.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    collinear = [c for c in upper.columns if (upper[c] > config.CORR_THRESHOLD).any()]
    if collinear:
        print(f"Dropped features with |r| > {config.CORR_THRESHOLD} against an earlier "
              f"feature (redundant, and they destabilise linear models): {collinear}")
        df = df.drop(columns=collinear)
    else:
        print(f"No feature pair exceeds |r| > {config.CORR_THRESHOLD} "
              f"(expected: V1-V28 are PCA components and therefore near-orthogonal).")
    return df


# ----------------------------------------------------------------------
# Scaling and splits
# ----------------------------------------------------------------------
def make_preprocessor():
    """Scale only the raw columns; V1..V28 are already on a comparable PCA scale.

    RobustScaler (median / IQR) is used because Amount is heavily right-skewed and
    fraud lives in the tails, so StandardScaler would be dragged around by exactly
    the outliers we care about.

    Scale matters for the proximity-based and gradient-based models: k-NN and the
    One-Class SVM compute Euclidean / RBF distances, so an unscaled Amount
    (0 - 25,000) would dominate every pairwise distance and the PCA features would
    stop contributing; Logistic Regression needs it to converge. Decision Tree,
    Random Forest, LightGBM and Gaussian Naive Bayes are scale-invariant (they
    split or model one feature at a time), so applying the same preprocessor to
    them is harmless and keeps a single pipeline.
    """
    pre = ColumnTransformer(
        [("robust", RobustScaler(), RAW_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def split_random(df):
    """Protocol (i): stratified random 70/15/15."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=config.TEST_SIZE + config.VAL_SIZE,
        stratify=y, random_state=config.RANDOM_STATE)
    rel_test = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp,
        random_state=config.RANDOM_STATE)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def split_time(df):
    """Protocol (ii): time-ordered 70/15/15 by `Time` - no shuffling, no stratifying."""
    ordered = df.sort_values("Time").reset_index(drop=True)
    n = len(ordered)
    cuts = {"train": slice(0, int(0.70 * n)),
            "val": slice(int(0.70 * n), int(0.85 * n)),
            "test": slice(int(0.85 * n), n)}
    return {name: (ordered.iloc[s].drop(columns=[TARGET]), ordered.iloc[s][TARGET])
            for name, s in cuts.items()}


def leakage_audit():
    """Task 1.1 deliverable: where test information could leak, and what stops it."""
    rows = [
        ("Scaling",
         "Fitting RobustScaler on all rows leaks the test median and IQR into training.",
         "The scaler is a step inside the imblearn Pipeline, so it is re-fitted on the "
         "training part of every CV fold and on train only for the final fit."),
        ("Oversampling (SMOTE)",
         "Synthesising before splitting puts interpolated copies of test frauds in train.",
         "SMOTE is a Pipeline step, so it runs after the fold split and never sees the "
         "validation fold."),
        ("Undersampling",
         "Resampling the whole set changes the test prior and inflates precision.",
         "RandomUnderSampler sits in the same Pipeline; validation and test keep their "
         "natural 0.17% fraud prior."),
        ("Feature selection",
         "Dropping features using the target over all rows is target leakage.",
         "Only target-free rules are used (zero variance, feature-feature |r|). The "
         "correlation-with-target ranking is used for plots only, never for removal."),
        ("Duplicates",
         "An exact duplicate split across train and test is a memorised answer.",
         "Duplicates are removed before any split is made."),
        ("Hyperparameters and thresholds",
         "Tuning on test turns it into a second validation set.",
         "Hyperparameters come from 5-fold CV on train; the decision threshold and the "
         "cost-optimal cut-off come from the validation split. Test is scored once."),
        ("Temporal order",
         "A random split lets the model learn from the future.",
         "Protocol (ii) trains strictly on the earliest 70% by Time, and Task 1.5 refits "
         "on that split instead of reusing the random-split model."),
    ]
    audit = pd.DataFrame(rows, columns=["Where", "How it could leak", "What prevents it"])
    print("\nLEAKAGE AUDIT")
    for _, r in audit.iterrows():
        print(f"  - {r['Where']}: {r['How it could leak']}")
        print(f"      -> {r['What prevents it']}")
    return audit


# ----------------------------------------------------------------------
def prepare_data(verbose=True):
    """Load -> profile -> clean -> split. Returns a `Dataset`."""
    if verbose:
        section("TASK 1.1 - Data understanding and preprocessing")
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"{config.DATA_PATH} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and put it there.")

    df = pd.read_csv(config.DATA_PATH)
    if config.SAMPLE_FRACTION < 1.0:
        # Stratified subsample so the fraud rate survives a smoke test.
        df = df.groupby(TARGET, group_keys=False).sample(
            frac=config.SAMPLE_FRACTION, random_state=config.RANDOM_STATE)
        print(f"[!] Running on a {config.SAMPLE_FRACTION:.0%} stratified subsample.")

    if verbose:
        profile(df)
        plot_top_features(df)
    df = clean(df)

    X_tr, y_tr, X_val, y_val, X_te, y_te = split_random(df)
    if verbose:
        print(f"\nRandom split -> train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
        print(f"Frauds per split: train {y_tr.sum()}, val {y_val.sum()}, test {y_te.sum()}")
        leakage_audit()

    return Dataset(X_tr, y_tr, X_val, y_val, X_te, y_te,
                   time_splits=split_time(df), features=list(X_tr.columns))


if __name__ == "__main__":
    prepare_data()
