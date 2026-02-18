"""
evaluate.py
===========
End-to-end evaluation pipeline.

Runs the custom Decision Tree and sklearn's DecisionTreeClassifier
on three datasets, computes metrics using metrics.py (from scratch),
and outputs:
  - A formatted console comparison table
  - A results.json file

Datasets are loaded via datasets.py (Section 3).
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.impute import SimpleImputer

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from decisionTree import DecisionTreeClassifier as CustomDT
from metrics import (
    evaluate,
    print_comparison_table,
    save_results_json,
    _run_sanity_checks,
)
from datasets import (
    load_real_dataset,
    load_synthetic_dataset,
    load_imbalanced_dataset,
)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


# ── Dataset loaders are in datasets.py (Section 3) ────────────────────────────


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate_dataset(X, y, dataset_name):
    """
    Split data, train both models, compute metrics, return result rows.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Custom Decision Tree ───────────────────────────
    custom = CustomDT(max_depth=10, min_samples_split=5,
                      missing_value_strategy='mean')
    custom.fit(X_train, y_train)
    y_pred_custom = custom.predict(X_test)

    custom_metrics = evaluate(y_test, y_pred_custom, average='macro')
    custom_row = {
        'dataset'  : dataset_name,
        'model'    : 'Custom DT',
        **custom_metrics,
    }

    # ── sklearn Decision Tree ──────────────────────────
    # Impute NaNs for sklearn (it doesn't handle them natively)
    imp = SimpleImputer(strategy='mean')
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp  = imp.transform(X_test)

    sk = SklearnDT(max_depth=10, random_state=RANDOM_STATE)
    sk.fit(X_train_imp, y_train)
    y_pred_sk = sk.predict(X_test_imp)

    sk_metrics = evaluate(y_test, y_pred_sk, average='macro')
    sk_row = {
        'dataset'  : dataset_name,
        'model'    : 'Sklearn DT',
        **sk_metrics,
    }

    return [custom_row, sk_row]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    # Step 1 – Sanity checks for metrics
    _run_sanity_checks()

    # Step 2 – Load datasets (via datasets.py)
    datasets = [
        load_real_dataset(),
        load_synthetic_dataset(),
        load_imbalanced_dataset(),
    ]

    # Step 3 – Evaluate
    all_results = []
    for X, y, name in datasets:
        print(f"\n[eval] Evaluating: {name}")
        rows = evaluate_dataset(X, y, name)
        all_results.extend(rows)

    # Step 4 – Display comparison table
    print_comparison_table(all_results)

    # Step 5 – Save to JSON
    save_results_json(all_results, path=RESULTS_PATH)


if __name__ == "__main__":
    main()
