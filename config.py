"""Central configuration. Edit here instead of inside the task scripts."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "creditcard.csv"   # Track A: ULB Credit Card Fraud (Kaggle)
FIG_DIR = ROOT / "figures"                     # all plots are written here
TABLE_DIR = ROOT / "tables"                    # all result tables (csv) are written here

# --- Reproducibility ---------------------------------------------------
RANDOM_STATE = 42

# --- Runtime knobs -----------------------------------------------------
# Fraction of rows to use. Keep < 1.0 for quick smoke tests, set to 1.0 for the
# final run that goes into the report.
SAMPLE_FRACTION = 1.0
N_JOBS = -1

# --- Protocol ----------------------------------------------------------
TEST_SIZE = 0.15          # 70 / 15 / 15 train / val / test
VAL_SIZE = 0.15
CV_FOLDS = 5
N_SEARCH_ITER = 8         # RandomizedSearchCV budget per model
CORR_THRESHOLD = 0.95     # |r| above which one of a feature pair is dropped

# --- Task 1.3 cost matrix ---------------------------------------------
# A missed fraud costs the transaction Amount; a false alarm costs a fixed
# investigation cost.
COST_FALSE_ALARM = 5.0

# --- Task 1.4 ----------------------------------------------------------
ANOMALY_SUBSAMPLE = 10_000   # LOF / One-Class SVM are O(n^2)-ish: fit on a subsample
                             # (kept modest so it's safe on free-tier Colab/Kaggle RAM)
LABEL_BUDGETS = (0.01, 0.05, 0.10, 0.25, 1.00)

# --- Task 1.5 ----------------------------------------------------------
N_TIME_WINDOWS = 4
