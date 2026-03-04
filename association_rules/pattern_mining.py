"""
pattern_mining.py
=================
Brute Force frequent itemset mining & association rule generation — from scratch.

Workflow:
  1. Discretize continuous features into categorical bins
  2. Enumerate ALL candidate itemsets (brute force)
  3. Count support and prune below min_support
  4. Generate association rules above min_confidence

No external mining libraries (mlxtend, etc.) are used.
"""

import numpy as np
import time
from itertools import combinations


# ── 1. Discretization ─────────────────────────────────────────────────────────

def discretize_data(X, y, n_bins=3):
    """
    Convert continuous feature matrix + labels into a list of transactions.

    Each sample becomes a set of items like:
        {"F0=Low", "F1=High", "F3=Med", ..., "Class=1"}

    Uses quantile-based (equal-frequency) binning so each bin has
    roughly the same number of samples.

    Parameters
    ----------
    X      : np.ndarray, shape (n_samples, n_features)
    y      : np.ndarray, shape (n_samples,)
    n_bins : int, number of bins per feature (default 3 → Low/Med/High)

    Returns
    -------
    transactions : list of frozensets, length n_samples
    all_items    : sorted list of every unique item across all transactions
    """
    bin_labels = {
        2: ["Low", "High"],
        3: ["Low", "Med", "High"],
        4: ["Low", "MedLow", "MedHigh", "High"],
        5: ["VLow", "Low", "Med", "High", "VHigh"],
    }
    labels = bin_labels.get(n_bins, [f"B{i}" for i in range(n_bins)])

    transactions = []
    n_samples, n_features = X.shape

    # Precompute bin edges per feature (quantile-based)
    bin_edges = []
    for col_idx in range(n_features):
        col = X[:, col_idx]
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(col, quantiles)
        # Ensure unique edges (can happen with many identical values)
        edges = np.unique(edges)
        bin_edges.append(edges)

    for i in range(n_samples):
        items = set()
        for col_idx in range(n_features):
            val = X[i, col_idx]
            edges = bin_edges[col_idx]
            # Determine bin index
            bin_idx = np.searchsorted(edges[1:], val, side="right")
            bin_idx = min(bin_idx, n_bins - 1)
            item_label = labels[min(bin_idx, len(labels) - 1)]
            items.add(f"F{col_idx}={item_label}")
        # Append class label
        items.add(f"Class={int(y[i])}")
        transactions.append(frozenset(items))

    # Collect all unique items
    all_items = sorted(set().union(*transactions))
    return transactions, all_items


# ── 2. Support Counting ───────────────────────────────────────────────────────

def count_support(transactions, itemset):
    """Count fraction of transactions that contain the itemset."""
    n = len(transactions)
    if n == 0:
        return 0.0
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / n


# ── 3. Brute Force Frequent Itemsets ───────────────────────────────────────────

def brute_force_frequent_itemsets(transactions, all_items, min_support=0.1,
                                   max_itemset_size=3):
    """
    Enumerate ALL possible itemsets up to max_itemset_size and keep those
    whose support >= min_support.

    Parameters
    ----------
    transactions     : list of frozensets
    all_items        : list of unique items
    min_support      : float, minimum support threshold
    max_itemset_size : int, max number of items in an itemset

    Returns
    -------
    frequent : dict {frozenset: float_support}
    candidates_checked : int, total number of candidates evaluated
    """
    frequent = {}
    candidates_checked = 0

    for size in range(1, max_itemset_size + 1):
        for combo in combinations(all_items, size):
            candidate = frozenset(combo)
            candidates_checked += 1
            sup = count_support(transactions, candidate)
            if sup >= min_support:
                frequent[candidate] = sup

        # Progress indicator for large searches
        if size <= max_itemset_size:
            count_at_size = sum(1 for k in frequent if len(k) == size)
            print(f"    [brute] Size {size}: "
                  f"{count_at_size} frequent itemsets found")

    return frequent, candidates_checked


# ── 4. Association Rule Generation ─────────────────────────────────────────────

def generate_rules(frequent_itemsets, transactions, min_confidence=0.5):
    """
    Generate association rules from frequent itemsets.

    For each frequent itemset of size >= 2, try every non-empty proper subset
    as the antecedent and the remainder as the consequent.

    Rule:  antecedent → consequent
      confidence = support(antecedent ∪ consequent) / support(antecedent)
      lift       = confidence / support(consequent)

    Parameters
    ----------
    frequent_itemsets : dict {frozenset: support}
    transactions      : list of frozensets
    min_confidence    : float

    Returns
    -------
    rules : list of dicts with keys:
            antecedent, consequent, support, confidence, lift
    """
    rules = []

    for itemset, sup in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        # Generate all non-empty proper subsets as antecedents
        for i in range(1, len(items)):
            for ante_tuple in combinations(items, i):
                antecedent = frozenset(ante_tuple)
                consequent = itemset - antecedent

                if len(consequent) == 0:
                    continue

                # Confidence = support(A∪C) / support(A)
                ante_sup = frequent_itemsets.get(antecedent, None)
                if ante_sup is None:
                    ante_sup = count_support(transactions, antecedent)
                if ante_sup == 0:
                    continue

                confidence = sup / ante_sup

                if confidence < min_confidence:
                    continue

                # Lift = confidence / support(C)
                cons_sup = frequent_itemsets.get(consequent, None)
                if cons_sup is None:
                    cons_sup = count_support(transactions, consequent)

                lift = confidence / cons_sup if cons_sup > 0 else 0.0

                rules.append({
                    "antecedent" : sorted(antecedent),
                    "consequent" : sorted(consequent),
                    "support"    : round(sup, 4),
                    "confidence" : round(confidence, 4),
                    "lift"       : round(lift, 4),
                })

    # Sort by confidence descending, then lift descending
    rules.sort(key=lambda r: (-r["confidence"], -r["lift"]))
    return rules


# ── 5. Orchestrator ────────────────────────────────────────────────────────────

def run_brute_force(X, y, dataset_name, min_support=0.1, min_confidence=0.5,
                     n_bins=3, max_itemset_size=3):
    """
    Full Brute Force pipeline: discretize → mine → generate rules.

    Returns
    -------
    result : dict with keys:
        dataset, algorithm, time_sec, n_frequent_itemsets,
        n_rules, candidates_checked, top_rules, frequent_itemsets
    """
    print(f"\n  [Brute Force] Mining: {dataset_name}")
    print(f"    min_support={min_support}, min_confidence={min_confidence}, "
          f"n_bins={n_bins}, max_k={max_itemset_size}")

    start = time.perf_counter()

    # Step 1: Discretize
    transactions, all_items = discretize_data(X, y, n_bins=n_bins)
    print(f"    Transactions: {len(transactions)}, Unique items: {len(all_items)}")

    # Step 2: Mine frequent itemsets (brute force)
    freq, candidates_checked = brute_force_frequent_itemsets(
        transactions, all_items,
        min_support=min_support,
        max_itemset_size=max_itemset_size,
    )

    # Step 3: Generate rules
    rules = generate_rules(freq, transactions, min_confidence=min_confidence)

    elapsed = time.perf_counter() - start

    print(f"    Frequent itemsets: {len(freq)}")
    print(f"    Rules generated : {len(rules)}")
    print(f"    Candidates checked: {candidates_checked}")
    print(f"    Time: {elapsed:.4f}s")

    return {
        "dataset"             : dataset_name,
        "algorithm"           : "Brute Force",
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
    result = run_brute_force(X, y, name, min_support=0.15, min_confidence=0.6)

    print(f"\n  Top 5 rules:")
    for i, rule in enumerate(result["top_rules"][:5], 1):
        print(f"    {i}. {rule['antecedent']} → {rule['consequent']}  "
              f"conf={rule['confidence']:.3f}  lift={rule['lift']:.3f}")
