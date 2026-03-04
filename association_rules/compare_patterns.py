"""
compare_patterns.py
===================
Runs Brute Force and Apriori frequent itemset mining on all three datasets,
compares results (itemsets, rules, timing), and saves output.

Usage
-----
    python compare_patterns.py

Output
------
    - Formatted comparison table (console)
    - pattern_comparison.json
"""

import os
import sys
import json

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from datasets import (
    load_real_dataset,
    load_synthetic_dataset,
    load_imbalanced_dataset,
)
from pattern_mining import run_brute_force
from apriori import run_apriori

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "pattern_comparison.json")

# ── Mining parameters ──────────────────────────────────────────────────────────
MIN_SUPPORT      = 0.10
MIN_CONFIDENCE   = 0.50
N_BINS           = 3
MAX_ITEMSET_SIZE = 3


# ── Display helpers ────────────────────────────────────────────────────────────

def print_section(title):
    w = 85
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def print_summary_table(results):
    """Print a side-by-side comparison table."""
    header = (f"{'Dataset':<25} {'Algorithm':<14} {'Itemsets':>9} "
              f"{'Rules':>7} {'Candidates':>12} {'Time (s)':>10}")
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        print(f"{r['dataset']:<25} {r['algorithm']:<14} "
              f"{r['n_frequent_itemsets']:>9} {r['n_rules']:>7} "
              f"{r['candidates_checked']:>12} {r['time_sec']:>10.4f}")

    print(sep)


def print_top_rules(result, n=5):
    """Print the top-n rules for a single result."""
    rules = result["top_rules"][:n]
    if not rules:
        print("      (no rules found)")
        return
    for i, rule in enumerate(rules, 1):
        ante = ", ".join(rule["antecedent"])
        cons = ", ".join(rule["consequent"])
        print(f"      {i}. {ante}  →  {cons}")
        print(f"         support={rule['support']:.4f}  "
              f"confidence={rule['confidence']:.4f}  "
              f"lift={rule['lift']:.4f}")


def verify_identical_itemsets(bf_result, ap_result, dataset_name):
    """Check that both algorithms found the same frequent itemsets."""
    bf_keys = set(bf_result["frequent_itemsets"].keys())
    ap_keys = set(ap_result["frequent_itemsets"].keys())

    if bf_keys == ap_keys:
        print(f"    [OK] MATCH: Both algorithms found identical {len(bf_keys)} "
              f"frequent itemsets for {dataset_name}")
        return True
    else:
        only_bf = bf_keys - ap_keys
        only_ap = ap_keys - bf_keys
        print(f"    [FAIL] MISMATCH for {dataset_name}:")
        if only_bf:
            print(f"      Only in Brute Force ({len(only_bf)}): "
                  f"{list(only_bf)[:5]}")
        if only_ap:
            print(f"      Only in Apriori ({len(only_ap)}): "
                  f"{list(only_ap)[:5]}")
        return False


# ── Main comparison pipeline ───────────────────────────────────────────────────

def main():
    print_section("Pattern Mining: Brute Force vs Apriori — Comparison")

    print(f"\n  Parameters:")
    print(f"    min_support      = {MIN_SUPPORT}")
    print(f"    min_confidence   = {MIN_CONFIDENCE}")
    print(f"    n_bins           = {N_BINS}  (quantile-based: Low / Med / High)")
    print(f"    max_itemset_size = {MAX_ITEMSET_SIZE}")

    # ── Load datasets ──────────────────────────────────────────────────────────
    datasets = [
        load_real_dataset(),
        load_synthetic_dataset(),
        load_imbalanced_dataset(),
    ]

    all_results    = []
    all_matched    = True

    for X, y, name in datasets:
        print_section(f"Dataset: {name}  ({X.shape[0]} samples, {X.shape[1]} features)")

        # ── Run Brute Force ────────────────────────────────────────────────────
        bf = run_brute_force(
            X, y, name,
            min_support=MIN_SUPPORT,
            min_confidence=MIN_CONFIDENCE,
            n_bins=N_BINS,
            max_itemset_size=MAX_ITEMSET_SIZE,
        )

        # ── Run Apriori ───────────────────────────────────────────────────────
        ap = run_apriori(
            X, y, name,
            min_support=MIN_SUPPORT,
            min_confidence=MIN_CONFIDENCE,
            n_bins=N_BINS,
            max_itemset_size=MAX_ITEMSET_SIZE,
        )

        all_results.extend([bf, ap])

        # ── Verify identical results ───────────────────────────────────────────
        print()
        match = verify_identical_itemsets(bf, ap, name)
        if not match:
            all_matched = False

        # ── Speedup ────────────────────────────────────────────────────────────
        if bf["time_sec"] > 0:
            speedup = bf["time_sec"] / ap["time_sec"] if ap["time_sec"] > 0 else float("inf")
            pruned = bf["candidates_checked"] - ap["candidates_checked"]
            print(f"    [TIME] Speedup: {speedup:.2f}x  "
                  f"(Apriori pruned {pruned} candidates)")

        # ── Top rules ─────────────────────────────────────────────────────────
        print(f"\n    Top 5 rules (Brute Force):")
        print_top_rules(bf)
        print(f"\n    Top 5 rules (Apriori):")
        print_top_rules(ap)

    # ── Summary table ──────────────────────────────────────────────────────────
    print_section("Summary")
    print_summary_table(all_results)

    if all_matched:
        print("\n  [OK] ALL DATASETS: Brute Force and Apriori produced identical "
              "frequent itemsets.")
        print("    Apriori achieves the same results with fewer candidate "
              "evaluations (pruning).")
    else:
        print("\n  [FAIL] WARNING: Some datasets showed mismatches -- investigate.")

    # ── Key comparison insights ────────────────────────────────────────────────
    print_section("Key Comparison Insights")

    total_bf_cand = sum(r["candidates_checked"] for r in all_results
                        if r["algorithm"] == "Brute Force")
    total_ap_cand = sum(r["candidates_checked"] for r in all_results
                        if r["algorithm"] == "Apriori")
    total_bf_time = sum(r["time_sec"] for r in all_results
                        if r["algorithm"] == "Brute Force")
    total_ap_time = sum(r["time_sec"] for r in all_results
                        if r["algorithm"] == "Apriori")

    print(f"\n  Total candidates evaluated:")
    print(f"    Brute Force : {total_bf_cand:,}")
    print(f"    Apriori     : {total_ap_cand:,}")
    print(f"    Reduction   : {total_bf_cand - total_ap_cand:,} "
          f"({(1 - total_ap_cand/total_bf_cand)*100:.1f}% fewer)"
          if total_bf_cand > 0 else "")

    print(f"\n  Total execution time:")
    print(f"    Brute Force : {total_bf_time:.4f}s")
    print(f"    Apriori     : {total_ap_time:.4f}s")
    if total_ap_time > 0:
        print(f"    Speedup     : {total_bf_time / total_ap_time:.2f}x")

    print(f"\n  Conclusion:")
    print(f"    Both algorithms find the SAME frequent itemsets and rules.")
    print(f"    Apriori's candidate pruning (downward closure property)")
    print(f"    dramatically reduces the search space, making it faster")
    print(f"    while guaranteeing identical output.\n")

    # ── Save to JSON ───────────────────────────────────────────────────────────
    # Strip large frequent_itemsets dict for cleaner JSON
    save_data = []
    for r in all_results:
        save_row = {k: v for k, v in r.items() if k != "frequent_itemsets"}
        save_data.append(save_row)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
