"""
datasets.py
===========
Section 3 — Standardized dataset loaders.

Provides three loader functions, each returning (X, y, dataset_name):
  - load_real_dataset()       : UCI Adult via fetch_openml (first 5 000 rows)
  - load_synthetic_dataset()  : make_classification with 10% label noise
  - load_imbalanced_dataset() : make_classification with 95/5 class split

All outputs:
  X  : np.ndarray of float64, shape (n_samples, n_features)
  y  : np.ndarray of int,     shape (n_samples,)
  name : str label used in reporting

Compatible with CustomDT, SklearnDT, and the metrics pipeline without
any further preprocessing.
"""

import numpy as np
from sklearn.datasets import make_classification

RANDOM_STATE = 42   # fixed seed for reproducibility


# ── Helpers ────────────────────────────────────────────────────────────────────

def _label_encode_column(col):
    """Map a string/object column to integer codes (0, 1, 2, …)."""
    unique_vals = sorted(set(str(v) for v in col))
    mapping = {v: i for i, v in enumerate(unique_vals)}
    return np.array([mapping[str(v)] for v in col], dtype=float)


# ── 1. Real-world dataset ──────────────────────────────────────────────────────

def load_real_dataset():
    """
    UCI Adult (income) dataset via sklearn fetch_openml(data_id=1590).

    - Source  : fetch_openml(data_id=1590)  — stable numeric ID, avoids
                fragile name/version resolution
    - Rows    : first 5 000 (after dropping NaN rows)
    - Features: 14 (mix of numeric + categorical → all encoded to float)
    - Target  : binary (0 = <=50K, 1 = >50K)

    Retry policy:
      Attempt 1 — fetch_openml(data_id=1590, as_frame=True)
      Attempt 2 — retry with parser='auto' on failure
      Fallback  — synthetic 5000×14 dataset if OpenML unreachable

    Returns
    -------
    X    : np.ndarray, shape (n, 14), dtype float64
    y    : np.ndarray, shape (n,),    dtype int
    name : "Real-world (Adult)" or "Real-world (Adult — fallback)"
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    def _process(frame):
        """Encode categoricals, split X/y, return numpy arrays."""
        df = frame.dropna().iloc[:5000].copy()
        X_df = df.drop(columns=["class"])
        y_raw = df["class"]
        X_enc = X_df.copy()
        for col in X_enc.columns:
            if X_enc[col].dtype.name in ("category", "object"):
                X_enc[col] = _label_encode_column(X_enc[col].values)
        X = X_enc.values.astype(float)
        y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
        return X, y

    # ── Attempt 1: data_id=1590, default parser ────────────────────────────
    try:
        print("[data] Fetching UCI Adult (data_id=1590) from OpenML ...")
        adult = fetch_openml(data_id=1590, as_frame=True)
        X, y = _process(adult.frame)
        print(f"[data] Real-world  : {X.shape[0]} samples, "
              f"{X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y, "Real-world (Adult)"

    except Exception as e1:
        print(f"[data] Attempt 1 failed: {e1}")

    # ── Attempt 2: retry with parser='auto' ────────────────────────────────
    try:
        print("[data] Retrying with parser='auto' ...")
        adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        X, y = _process(adult.frame)
        print(f"[data] Real-world  : {X.shape[0]} samples, "
              f"{X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y, "Real-world (Adult)"

    except Exception as e2:
        print(f"[data] Attempt 2 failed: {e2}")

    # ── Fallback: synthetic stand-in (OpenML unreachable) ─────────────────
    print("[data] WARNING: OpenML unreachable. Using synthetic fallback "
          "(5000 samples, 14 features).")
    X, y = make_classification(
        n_samples=5000,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        random_state=RANDOM_STATE,
    )
    X = X.astype(float)
    y = y.astype(int)
    print(f"[data] Real-world* : {X.shape[0]} samples, "
          f"{X.shape[1]} features, {len(np.unique(y))} classes  [FALLBACK]")
    return X, y, "Real-world (Adult — fallback)"


# ── 2. Synthetic dataset ───────────────────────────────────────────────────────

def load_synthetic_dataset():
    """
    Programmatically generated dataset with injected label noise.

    Parameters (fixed):
      n_samples    = 2000
      n_features   = 15
      n_informative= 10
      flip_y       = 0.10   (10 % label noise)
      random_state = RANDOM_STATE

    Returns
    -------
    X    : np.ndarray, shape (2000, 15), dtype float64
    y    : np.ndarray, shape (2000,),    dtype int
    name : "Synthetic (noisy)"
    """
    NAME = "Synthetic (noisy)"

    print("[data] Generating synthetic dataset ...")
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        flip_y=0.10,            # inject 10 % label noise
        random_state=RANDOM_STATE,
    )
    X = X.astype(float)
    y = y.astype(int)

    print(f"[data] Synthetic   : {X.shape[0]} samples, "
          f"{X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y, NAME


# ── 3. Imbalanced dataset ──────────────────────────────────────────────────────

def load_imbalanced_dataset():
    """
    Highly imbalanced binary dataset: 95 % class 0, 5 % class 1.

    Parameters (fixed):
      n_samples    = 2000
      n_features   = 12
      weights      = [0.95, 0.05]
      flip_y       = 0.0   (no extra noise; imbalance is the challenge)
      random_state = RANDOM_STATE

    Returns
    -------
    X    : np.ndarray, shape (2000, 12), dtype float64
    y    : np.ndarray, shape (2000,),    dtype int
    name : "Imbalanced (95/5)"
    """
    NAME = "Imbalanced (95/5)"

    print("[data] Generating imbalanced dataset ...")
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.95, 0.05],   # 95 % majority / 5 % minority
        flip_y=0.0,
        random_state=RANDOM_STATE,
    )
    X = X.astype(float)
    y = y.astype(int)

    counts = np.bincount(y)
    print(f"[data] Imbalanced  : {X.shape[0]} samples, "
          f"{X.shape[1]} features, class dist = {dict(enumerate(counts))}")
    return X, y, NAME


# ── Validation (run as script) ─────────────────────────────────────────────────

def _validate_dataset(X, y, name):
    """Basic shape and type checks."""
    assert isinstance(X, np.ndarray),          f"{name}: X must be ndarray"
    assert isinstance(y, np.ndarray),          f"{name}: y must be ndarray"
    assert X.ndim == 2,                        f"{name}: X must be 2-D"
    assert y.ndim == 1,                        f"{name}: y must be 1-D"
    assert X.shape[0] == y.shape[0],           f"{name}: row count mismatch"
    assert X.dtype == float,                   f"{name}: X must be float"
    assert np.issubdtype(y.dtype, np.integer), f"{name}: y must be integer"
    assert X.shape[0] >= 1000,                 f"{name}: need >= 1000 samples"
    assert X.shape[1] >= 10,                   f"{name}: need >= 10 features"
    print(f"  PASS  {name}  shape=({X.shape[0]}, {X.shape[1]})  "
          f"classes={sorted(np.unique(y).tolist())}")


if __name__ == "__main__":
    print("=" * 55)
    print("DATASET VALIDATION")
    print("=" * 55)

    for loader in [load_real_dataset, load_synthetic_dataset, load_imbalanced_dataset]:
        X, y, name = loader()
        _validate_dataset(X, y, name)
        print()

    print("=" * 55)
    print("ALL DATASET VALIDATIONS PASSED")
    print("=" * 55)
