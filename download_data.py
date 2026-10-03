"""Fetches the Track A dataset (ULB Credit Card Fraud, via Kaggle) automatically.

The raw CSV (~144 MB) is never committed to the repo (see .gitignore); this
module gets it onto disk at `config.DATA_PATH` instead. Resolution order,
cheapest and most reliable first:

  1. `config.DATA_PATH` already exists -> nothing to do.
  2. Running on Kaggle with the dataset attached as a notebook Input (mounted
     read-only under /kaggle/input) -> copy it in, no network call needed.
  3. `kagglehub.dataset_download(...)` -> downloads via the Kaggle API. This
     needs zero setup on Kaggle itself (the notebook environment is already
     authenticated); run locally it needs a Kaggle API token (see the
     instructions raised in the error message below).

We deliberately do NOT use sklearn's `fetch_openml(data_id=1597)` mirror of this
dataset: it is missing the `Time` column, which the assignment's time-ordered
split (Task 1.1) and drift analysis (Task 1.5) both depend on.
"""
import shutil
from pathlib import Path

import config

KAGGLE_HANDLE = "mlg-ulb/creditcardfraud"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def _find_on_kaggle_input():
    """Fast path: the dataset is already mounted because it was added as a Kaggle Input."""
    if not KAGGLE_INPUT_ROOT.exists():
        return None
    matches = list(KAGGLE_INPUT_ROOT.glob("**/creditcard.csv"))
    return matches[0] if matches else None


def _download_with_kagglehub():
    """Network path: pull the dataset through the Kaggle API."""
    import kagglehub  # imported lazily - only needed on this code path

    data_dir = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    matches = list(data_dir.glob("**/creditcard.csv"))
    if not matches:
        raise FileNotFoundError(f"kagglehub downloaded {data_dir} but it contains no creditcard.csv.")
    return matches[0]


def ensure_dataset():
    """Guarantee `config.DATA_PATH` exists on disk, downloading it if necessary."""
    if config.DATA_PATH.exists():
        return config.DATA_PATH

    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    found = _find_on_kaggle_input()
    if found:
        print(f"[data] found the dataset already attached at {found}, copying in...")
    else:
        print(f"[data] {config.DATA_PATH} not found - downloading '{KAGGLE_HANDLE}' via kagglehub...")
        try:
            found = _download_with_kagglehub()
        except Exception as e:
            raise RuntimeError(
                "Could not obtain creditcard.csv automatically.\n"
                f"  Underlying error: {e}\n"
                "Fix options:\n"
                "  1) On Kaggle: notebook sidebar -> Add Input -> search 'Credit Card Fraud "
                "Detection' (by mlg-ulb) -> add it, then re-run main.py.\n"
                "  2) Locally: create a Kaggle API token at kaggle.com/settings -> "
                "'Create New Token', save the downloaded kaggle.json to ~/.kaggle/kaggle.json "
                "(or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables), then re-run.\n"
                "  3) Or download creditcard.csv by hand from "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at "
                f"{config.DATA_PATH}."
            ) from e

    shutil.copy(found, config.DATA_PATH)
    print(f"[data] ready at {config.DATA_PATH} ({config.DATA_PATH.stat().st_size / 1e6:.1f} MB)")
    return config.DATA_PATH


if __name__ == "__main__":
    ensure_dataset()
