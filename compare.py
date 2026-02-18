"""
compare.py
==========
Section 4 — sklearn vs Custom Decision Tree: Side-by-Side Comparison.

Reuses dataset loaders and the evaluate_dataset() function from evaluate.py.
Metrics are computed exclusively via metrics.py (no sklearn.metrics).

Output
------
  - Formatted comparison table printed to the terminal
  - comparison.json saved alongside this file

Usage
-----
    python compare.py
"""

import os
import sys
import json

# ── Local imports (reuse existing pipeline components) ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import (          # shared split/train/eval helper
    evaluate_dataset,
    RANDOM_STATE,
    TEST_SIZE,
)
from datasets import (          # Section 3 dataset loaders
    load_real_dataset,
    load_synthetic_dataset,
    load_imbalanced_dataset,
)
from metrics import print_comparison_table   # formatted table printer

COMPARISON_PATH = os.path.join(os.path.dirname(__file__), "comparison.json")


# ── Output helpers ─────────────────────────────────────────────────────────────

def save_comparison_json(results, path=COMPARISON_PATH):
    """Persist the comparison results list to a JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[compare] Comparison saved -> {path}")


def print_section_header(title):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── Main comparison pipeline ───────────────────────────────────────────────────

def main():
    print_section_header("sklearn vs Custom Decision Tree — Side-by-Side Comparison")

    print(f"\n  Settings: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    print("  Metrics : Accuracy | Precision (macro) | Recall (macro) | F1 (macro)")
    print("  Metrics computed from scratch via metrics.py — no sklearn.metrics\n")

    # ── Load all three datasets ────────────────────────────────────────────────
    datasets = [
        load_real_dataset(),       # 5 000 samples, 14 features (Adult / fallback)
        load_synthetic_dataset(),  # 2 000 samples, 15 features, noise=10%
        load_imbalanced_dataset(), # 2 000 samples, 12 features, 95/5 split
    ]

    # ── Evaluate each dataset with both models ─────────────────────────────────
    all_results = []

    for X, y, name in datasets:
        print(f"\n[compare] Running: {name}")

        # evaluate_dataset() uses the SAME train_test_split for both models
        rows = evaluate_dataset(X, y, name)   # returns [custom_row, sklearn_row]
        all_results.extend(rows)

    # ── Print comparison table ─────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Persist to JSON ────────────────────────────────────────────────────────
    save_comparison_json(all_results)


if __name__ == "__main__":
    main()
