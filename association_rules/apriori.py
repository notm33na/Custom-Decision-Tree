"""
apriori.py
==========
Apriori frequent itemset mining — from scratch.

Uses the downward closure (anti-monotone) property:
  "All subsets of a frequent itemset must also be frequent."

This allows pruning candidates before counting support, making it
far more efficient than brute force for large item spaces.

Reuses discretize_data() and generate_rules() from pattern_mining.py.
"""

import numpy as np
import time
from itertools import combinations

# Reuse shared utilities from pattern_mining
from pattern_mining import (
    discretize_data,
    count_support,
    generate_rules,
)


# ── 1. Apriori Candidate Generation ───────────────────────────────────────────

def _apriori_gen(prev_frequent_sets, k):
    """
    Generate size-k candidate itemsets from size-(k-1) frequent itemsets
    using the F(k-1) × F(k-1) join method.

    Join rule:
      Two itemsets of size k-1 are joined if they share the first k-2 items
      (when sorted).

    Prune rule (downward closure):
      A candidate is pruned if any of its (k-1)-subsets is NOT in
      prev_frequent_sets.

    Parameters
    ----------
    prev_frequent_sets : set of frozensets, each of size k-1
    k                  : int, desired candidate size

    Returns
    -------
    candidates : set of frozensets, each of size k
    """
    candidates = set()
    prev_list = [sorted(fs) for fs in prev_frequent_sets]
    prev_set  = prev_frequent_sets  # for fast subset checking

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            # Join: check if first k-2 items match
            l1 = prev_list[i]
            l2 = prev_list[j]

            if l1[:k - 2] == l2[:k - 2]:
                candidate = frozenset(l1) | frozenset(l2)

                if len(candidate) == k:
                    # Prune: check all (k-1)-subsets are frequent
                    all_subsets_frequent = True
                    for sub in combinations(candidate, k - 1):
                        if frozenset(sub) not in prev_set:
                            all_subsets_frequent = False
                            break

                    if all_subsets_frequent:
                        candidates.add(candidate)

    return candidates


# ── 2. Apriori Algorithm ──────────────────────────────────────────────────────

def apriori_frequent_itemsets(transactions, all_items, min_support=0.1,
                                max_itemset_size=3):
    """
    Classic Apriori: level-wise bottom-up frequent itemset mining.

    Level 1: Count support for every single item.
    Level k: Generate candidates from level k-1 frequent sets,
             count support, prune below threshold.

    Parameters
    ----------
    transactions     : list of frozensets
    all_items        : list of unique items
    min_support      : float
    max_itemset_size : int

    Returns
    -------
    frequent         : dict {frozenset: float_support}
    candidates_checked : int, total candidates evaluated
    """
    frequent = {}
    candidates_checked = 0

    # ── Level 1: single items ─────────────────────────────
    current_frequent = set()
    for item in all_items:
        candidate = frozenset([item])
        candidates_checked += 1
        sup = count_support(transactions, candidate)
        if sup >= min_support:
            frequent[candidate] = sup
            current_frequent.add(candidate)

    count_1 = len(current_frequent)
    print(f"    [apriori] Size 1: {count_1} frequent itemsets found")

    if count_1 == 0:
        return frequent, candidates_checked

    # ── Levels 2 .. max_itemset_size ──────────────────────
    k = 2
    while k <= max_itemset_size and len(current_frequent) > 0:
        # Generate candidates using Apriori-gen
        candidates = _apriori_gen(current_frequent, k)

        next_frequent = set()
        for candidate in candidates:
            candidates_checked += 1
            sup = count_support(transactions, candidate)
            if sup >= min_support:
                frequent[candidate] = sup
                next_frequent.add(candidate)

        print(f"    [apriori] Size {k}: "
              f"{len(next_frequent)} frequent itemsets found "
              f"(from {len(candidates)} candidates)")

        current_frequent = next_frequent
        k += 1

    return frequent, candidates_checked


# ── 3. Orchestrator ────────────────────────────────────────────────────────────

def run_apriori(X, y, dataset_name, min_support=0.1, min_confidence=0.5,
                n_bins=3, max_itemset_size=3):
    """
    Full Apriori pipeline: discretize → mine → generate rules.

    Returns
    -------
    result : dict with keys:
        dataset, algorithm, time_sec, n_frequent_itemsets,
        n_rules, candidates_checked, top_rules, frequent_itemsets
    """
    print(f"\n  [Apriori] Mining: {dataset_name}")
    print(f"    min_support={min_support}, min_confidence={min_confidence}, "
          f"n_bins={n_bins}, max_k={max_itemset_size}")

    start = time.perf_counter()

    # Step 1: Discretize (same function as brute force → same bins)
    transactions, all_items = discretize_data(X, y, n_bins=n_bins)
    print(f"    Transactions: {len(transactions)}, Unique items: {len(all_items)}")

    # Step 2: Mine frequent itemsets (Apriori)
    freq, candidates_checked = apriori_frequent_itemsets(
        transactions, all_items,
        min_support=min_support,
        max_itemset_size=max_itemset_size,
    )

    # Step 3: Generate rules (same function as brute force)
    rules = generate_rules(freq, transactions, min_confidence=min_confidence)

    elapsed = time.perf_counter() - start

    print(f"    Frequent itemsets: {len(freq)}")
    print(f"    Rules generated : {len(rules)}")
    print(f"    Candidates checked: {candidates_checked}")
    print(f"    Time: {elapsed:.4f}s")

    return {
        "dataset"             : dataset_name,
        "algorithm"           : "Apriori",
        "time_sec"            : round(elapsed, 4),
        "n_frequent_itemsets" : len(freq),
        "n_rules"             : len(rules),
        "candidates_checked"  : candidates_checked,
        "top_rules"           : rules[:10],
        "frequent_itemsets"   : {str(sorted(k)): v for k, v in freq.items()},
    }


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datasets import load_synthetic_dataset

    X, y, name = load_synthetic_dataset()
    result = run_apriori(X, y, name, min_support=0.15, min_confidence=0.6)

    print(f"\n  Top 5 rules:")
    for i, rule in enumerate(result["top_rules"][:5], 1):
        print(f"    {i}. {rule['antecedent']} → {rule['consequent']}  "
              f"conf={rule['confidence']:.3f}  lift={rule['lift']:.3f}")
