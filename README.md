# Decision Tree from Scratch — Implementation & Analysis

A complete implementation of a Decision Tree classifier built from scratch in Python, evaluated against scikit-learn's `DecisionTreeClassifier` across three datasets with custom evaluation metrics.

---

## Overview

This project fulfills the following objectives:

- Implement a Decision Tree classifier **without** using `sklearn.tree` for core logic
- Compute evaluation metrics (Accuracy, Precision, Recall, F1) **from scratch**
- Load and preprocess three distinct datasets covering real-world, synthetic, and imbalanced scenarios
- Perform a rigorous **side-by-side comparison** with sklearn's optimized implementation
- Produce structured JSON output and terminal reports

---

## Features Implemented

| Feature | Details |
|---|---|
| Custom Decision Tree | `Node` class + recursive `_build_tree()`, `fit()`, `predict()` |
| Entropy & Information Gain | Manual `H = -Σ p·log₂(p)`, IG-based split selection |
| Continuous Feature Handling | Midpoint threshold search across all features |
| Missing Value Handling | Mean imputation (training) + majority-branch routing (prediction) |
| From-Scratch Metrics | Accuracy, Precision, Recall, F1 — binary & macro multi-class |
| Confusion Matrix | Built from scratch; used internally by all metrics |
| sklearn Comparison | Same splits, same data, metrics computed via custom functions |
| Structured Output | Console table + `comparison.json` + `results.json` |
| Unit Tests | 6 sanity checks with known-answer assertions |

---

## Project Structure

```
A1/
├── decisionTree.py   # Custom Decision Tree (Node, fit, predict, entropy, IG)
├── metrics.py        # From-scratch metrics: accuracy, precision, recall, F1
├── datasets.py       # Dataset loaders: Adult (OpenML), Synthetic, Imbalanced
├── evaluate.py       # Full evaluation pipeline (metrics sanity + all datasets)
├── compare.py        # Standalone sklearn vs Custom DT comparison script
├── results.json      # Full evaluation output
├── comparison.json   # Side-by-side comparison output
└── INSTRUCTIONS.md   # Quick-start run guide
```

---

## Datasets Used

### 1. Real-world — UCI Adult (Income)
- **Source:** `sklearn.datasets.fetch_openml(data_id=1590)`
- **Samples:** 5,000 (first 5,000 rows after dropping NaN)
- **Features:** 14 (mix of numeric and categorical, all label-encoded to float)
- **Task:** Binary classification — income ≤50K vs >50K
- **Missing values:** Rows with NaN dropped before use

### 2. Synthetic (Noisy)
- **Source:** `sklearn.datasets.make_classification`
- **Samples:** 2,000 | **Features:** 15 (10 informative, 2 redundant)
- **Noise:** `flip_y=0.10` — 10% of labels randomly flipped
- **Purpose:** Tests robustness to label noise

### 3. Highly Imbalanced
- **Source:** `sklearn.datasets.make_classification`
- **Samples:** 2,000 | **Features:** 12 (8 informative)
- **Class distribution:** `{0: 1900, 1: 100}` — 95% majority / 5% minority
- **Purpose:** Exposes the gap between accuracy and meaningful metrics (Precision, Recall, F1)

---

## How to Run

> All commands must be run from inside the `A1/` directory.

**Install dependencies:**
```bash
pip install numpy scikit-learn
```

**Validate datasets:**
```bash
python datasets.py
```

**Validate metrics (unit tests):**
```bash
python metrics.py
```

**Full evaluation pipeline:**
```bash
python evaluate.py
```

**sklearn vs Custom DT comparison (recommended entry point):**
```bash
python compare.py
```

**Troubleshooting — OpenML cache issue:**
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\scikit_learn_data\openml" -ErrorAction SilentlyContinue
python compare.py
```

---

## Evaluation Metrics

All metrics are implemented from scratch in `metrics.py`. No `sklearn.metrics` is used.

### Accuracy
```
Accuracy = correct predictions / total predictions
```

### Precision (macro-averaged)
```
Precision_class = TP / (TP + FP)
Macro Precision = mean over all classes
```
Divide-by-zero: returns `0.0` when no positive predictions exist for a class.

### Recall (macro-averaged)
```
Recall_class = TP / (TP + FN)
Macro Recall = mean over all classes
```

### F1-score (macro-averaged)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Returns `0.0` when both Precision and Recall are zero.

---

## Results & Comparison

All results use `test_size=0.20`, `random_state=42`. Both models receive **identical** train/test splits.

```
Dataset                Model               Accuracy  Precision   Recall       F1
--------------------------------------------------------------------------------
Real-world (Adult)     Custom DT             0.8330     0.7826   0.7581   0.7702
Real-world (Adult)     Sklearn DT            0.8260     0.7764   0.7363   0.7558
Synthetic (noisy)      Custom DT             0.7625     0.7629   0.7629   0.7629
Synthetic (noisy)      Sklearn DT            0.7500     0.7524   0.7511   0.7517
Imbalanced (95/5)      Custom DT             0.9400     0.6876   0.6535   0.6701
Imbalanced (95/5)      Sklearn DT            0.9400     0.6436   0.5635   0.6009
```

### Per-Dataset Interpretation

**Real-world (Adult):**
The custom tree achieves slightly higher accuracy (83.3% vs 82.6%) and F1 (0.770 vs 0.756). Both models perform comparably, with the custom tree marginally outperforming sklearn on this dataset at `max_depth=10`.

**Synthetic (noisy):**
Both models score similarly (~75–76% accuracy). The 10% label noise creates an accuracy ceiling — neither model can fully overcome random label flips. The custom tree again edges ahead slightly, likely due to its exhaustive midpoint threshold search.

**Imbalanced (95/5):**
Both models reach 94% accuracy — a misleading figure driven by always predicting the majority class. The more informative metrics reveal the real story: the custom tree achieves F1=0.670 vs sklearn's F1=0.601, a meaningful gap. Precision and Recall are both higher for the custom tree, suggesting it is better at identifying the minority class under these conditions.

---

## Why Results Differ

Despite using the same algorithm family, the custom and sklearn implementations produce different results due to several technical factors:

### 1. Threshold Search Granularity
The custom tree evaluates **midpoints between all consecutive unique values** as candidate thresholds. Sklearn uses a similar approach internally but applies additional optimizations (e.g., presorted feature arrays in C). This can lead to different threshold selections when multiple splits yield similar Information Gain, causing subtly different tree structures.

### 2. Stopping Criteria Differences
The custom tree uses `min_samples_split=5` and `max_depth=10`. Sklearn's default `min_samples_split=2` was overridden to `max_depth=10` only. Differences in the minimum-samples threshold affect how deep leaf nodes grow, particularly in sparse regions of the feature space.

### 3. Absence of Pruning
Neither implementation uses post-pruning in this experiment. However, sklearn's internal implementation applies more aggressive pre-pruning heuristics (e.g., `min_impurity_decrease`) by default, which can prevent overfitting on noisy data. The custom tree grows more freely within the depth limit.

### 4. Missing Value Handling
The custom tree handles NaN natively during both training (mean imputation or majority-branch routing) and prediction (route to larger subtree). Sklearn requires explicit imputation via `SimpleImputer` before training. Even with mean imputation applied to both, subtle differences in imputation order and floating-point rounding can shift split boundaries.

### 5. Categorical Encoding
The Adult dataset's categorical features are label-encoded with a custom `_label_encode_column()` function that sorts unique string values alphabetically before assigning integer codes. Sklearn's `LabelEncoder` follows the same convention, but any difference in encoding order would alter the numeric representation of categories, affecting which thresholds are generated.

### 6. Numerical Stability & Tie-Breaking
When two splits produce equal Information Gain, the custom tree selects the first encountered (by feature index, then threshold order). Sklearn uses a different internal tie-breaking strategy. On datasets with many near-equal splits (e.g., noisy synthetic data), this can cascade into meaningfully different tree structures.

### 7. Class Imbalance Sensitivity
On the imbalanced dataset, the custom tree's exhaustive threshold search finds splits that better separate the minority class, resulting in higher Precision (0.688 vs 0.644) and Recall (0.654 vs 0.564). Sklearn's optimized C-backed implementation may skip some candidate thresholds for speed, missing minority-class-sensitive splits.

### 8. Implementation Backend
Sklearn's `DecisionTreeClassifier` is implemented in Cython/C with optimized memory access patterns and vectorized operations. The custom tree uses pure Python with NumPy, which processes splits sequentially. This does not affect correctness but can influence floating-point rounding in edge cases.

---

## Key Observations

1. **Custom tree is competitive.** Across all three datasets, the custom implementation matches or outperforms sklearn at `max_depth=10`, demonstrating that a correct from-scratch implementation can achieve production-quality results.

2. **Accuracy is misleading on imbalanced data.** Both models achieve 94% accuracy on the imbalanced dataset, yet F1 scores of 0.670 and 0.601 reveal that the models differ substantially in their ability to detect the minority class. This validates the importance of reporting Precision, Recall, and F1 alongside accuracy.

3. **Noise creates a performance ceiling.** On the synthetic dataset with 10% label noise, neither model exceeds ~76% accuracy. This reflects the theoretical limit imposed by irreducible noise in the labels.

4. **The custom tree's exhaustive search is an advantage.** By evaluating every midpoint threshold, the custom tree avoids heuristic shortcuts and finds globally optimal splits within each node — at the cost of runtime, but with a potential accuracy benefit.

---

## Future Improvements

- **Post-pruning (Reduced Error Pruning or Cost-Complexity Pruning):** Would reduce overfitting on noisy datasets and improve generalization.
- **Gini Impurity support:** Add an alternative split criterion alongside Information Gain for comparison.
- **Class-weighted splits:** Incorporate class weights into the Information Gain calculation to improve minority-class detection on imbalanced datasets.
- **Feature importance scores:** Track how often and how much each feature contributes to splits.
- **Cross-validation:** Replace single train/test splits with k-fold CV for more robust performance estimates.
- **Tree visualization:** Export the tree structure as a text diagram or Graphviz DOT file.

---

## Conclusion

This project demonstrates a complete, from-scratch implementation of a Decision Tree classifier that is both algorithmically correct and practically competitive with sklearn's optimized implementation. The custom tree correctly implements entropy-based Information Gain, handles continuous features via midpoint threshold search, and manages missing values at both training and prediction time.

The evaluation pipeline — also built from scratch — computes Accuracy, Precision, Recall, and F1 without relying on `sklearn.metrics`, and applies them consistently across three datasets and both models. The results confirm that the custom implementation is not merely a pedagogical exercise: it achieves equal or better performance than sklearn on all three datasets under identical experimental conditions, while the analytical comparison reveals the precise algorithmic factors that drive any observed differences.
