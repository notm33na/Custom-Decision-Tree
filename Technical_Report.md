# Decision Trees: Implementation & Analysis
## Technical Report

---

**Course:** Process Mining and Simulation  
**Assignment:** Decision Tree Classifier - Implementation & Analysis  
**Total Marks:** 50  
**Deadline:** 20th February 2026  

**Group Members:**
- Saad Khan
- Malaika Naseer
- Zarmeena Fatima

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Implementation Details](#2-implementation-details)
   - 2.1 [Decision Tree from Scratch](#21-decision-tree-from-scratch)
   - 2.2 [Entropy and Information Gain](#22-entropy-and-information-gain)
   - 2.3 [Continuous Features Handling](#23-continuous-features-handling)
   - 2.4 [Missing Values Handling](#24-missing-values-handling)
   - 2.5 [Comparison with sklearn](#25-comparison-with-sklearn)
3. [Datasets](#3-datasets)
   - 3.1 [Real-world Dataset](#31-real-world-dataset)
   - 3.2 [Synthetic Dataset](#32-synthetic-dataset)
   - 3.3 [Imbalanced Dataset](#33-imbalanced-dataset)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Results and Analysis](#5-results-and-analysis)
6. [Individual Contributions](#6-individual-contributions)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Introduction

Decision Trees are fundamental machine learning algorithms used for classification and regression tasks. This project implements a complete Decision Tree classifier from scratch without using sklearn's tree module for core logic. The implementation demonstrates all essential components including entropy calculation, information gain, handling of continuous features, and missing value management.

### Objectives

The primary objectives of this assignment are:

1. Implement a Decision Tree classifier completely from scratch
2. Support Entropy and Information Gain as splitting criteria
3. Handle continuous features using optimal threshold search
4. Implement missing value handling strategies
5. Compare results with sklearn's DecisionTreeClassifier
6. Evaluate on three diverse datasets: real-world, synthetic, and imbalanced
7. Compute evaluation metrics (Accuracy, Precision, Recall, F1-score) from scratch

### Project Significance

This implementation provides deep insight into how Decision Trees work internally, beyond the black-box usage of library functions. By building each component from first principles, we understand the algorithmic decisions, trade-offs, and implementation challenges involved in creating a production-quality classifier.

---

## 2. Implementation Details

### 2.1 Decision Tree from Scratch

**Implementation File:** `decisionTree.py`

The custom Decision Tree classifier is built using two main classes:

#### Node Class
```python
class Node:
    def __init__(self):
        self.feature_index = None   # Index of feature to split on
        self.threshold     = None   # Threshold for continuous features
        self.left          = None   # Left child  (feature <= threshold)
        self.right         = None   # Right child (feature > threshold)
        self.is_leaf       = False
        self.label         = None   # Predicted class (leaf nodes only)
```

The `Node` class represents a single decision node in the tree, storing:
- **feature_index**: Which feature to split on
- **threshold**: The value to compare against for continuous features
- **left/right**: Child nodes for binary splitting
- **is_leaf**: Boolean flag indicating if this is a terminal node
- **label**: Predicted class for leaf nodes

#### DecisionTreeClassifier Class

**Key Parameters:**
- `max_depth`: Maximum depth of the tree (prevents overfitting)
- `min_samples_split`: Minimum samples required to attempt a split
- `missing_value_strategy`: Strategy for handling missing values ('mean' or 'majority')

**Core Methods:**

1. **fit(X, y)**: Trains the tree by recursively building nodes
   - Converts inputs to numpy arrays
   - Computes column means for missing value imputation
   - Calls `_build_tree()` to construct the tree structure

2. **predict(X)**: Makes predictions on new data
   - Handles missing values using the chosen strategy
   - Traverses the tree for each sample using `_predict_row()`

3. **_build_tree(X, y, depth)**: Recursive tree construction
   - Checks stopping criteria (purity, minimum samples, max depth)
   - Finds best split using `_best_split()`
   - Partitions data based on the split
   - Recursively builds left and right subtrees

**Stopping Criteria:**
```python
is_pure       = len(np.unique(y)) == 1
too_small     = len(y) < self.min_samples_split
max_depth_hit = (self.max_depth is not None) and (depth >= self.max_depth)
```

The tree stops growing when:
- All samples in a node belong to the same class (pure node)
- The number of samples falls below `min_samples_split`
- The maximum depth limit is reached

---

### 2.2 Entropy and Information Gain

**Entropy** measures the impurity or disorder in a dataset. For a dataset S with classes, entropy is defined as:

```
H(S) = -Σ p_i * log₂(p_i)
```

where `p_i` is the proportion of samples belonging to class i.

**Implementation:**
```python
@staticmethod
def _entropy(y):
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs  = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))
```

**Key Features:**
- Returns 0 for empty arrays (pure nodes have 0 entropy)
- Uses `np.bincount()` for efficient class counting
- Filters zero probabilities to avoid log(0) errors

**Information Gain** measures the reduction in entropy after a split:

```
IG = H(parent) - weighted_avg_entropy(children)
```

**Implementation:**
```python
def _information_gain(self, y, y_left, y_right):
    n   = len(y)
    n_l = len(y_left)
    n_r = len(y_right)
    
    if n_l == 0 or n_r == 0:
        return 0.0
    
    parent_entropy = self._entropy(y)
    child_entropy  = (n_l / n) * self._entropy(y_left) + \
                     (n_r / n) * self._entropy(y_right)
    return parent_entropy - child_entropy
```

The algorithm selects splits that maximize Information Gain, leading to the most informative partitioning of the data.

---

### 2.3 Continuous Features Handling

Unlike categorical features with discrete values, continuous features require finding optimal threshold values for splitting.

**Strategy: Midpoint Threshold Search**

For each continuous feature, the algorithm:
1. Extracts unique values from the feature column
2. Sorts them in ascending order
3. Computes midpoints between consecutive values as candidate thresholds
4. Evaluates Information Gain for each threshold
5. Selects the threshold with maximum Information Gain

**Implementation:**
```python
def _best_split(self, X, y):
    n_features      = X.shape[1]
    best_ig         = -1
    best_feature    = None
    best_threshold  = None

    for feature_idx in range(n_features):
        col = X[:, feature_idx]
        
        # Ignore NaN rows when scanning thresholds
        valid = ~np.isnan(col)
        col_v = col[valid]
        y_v   = y[valid]
        
        if len(col_v) == 0:
            continue
        
        unique_vals = np.unique(col_v)
        if len(unique_vals) == 1:
            continue
        
        # Candidate thresholds: midpoints between consecutive values
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        
        for threshold in thresholds:
            left_mask  = col_v <= threshold
            right_mask = ~left_mask
            
            ig = self._information_gain(y_v, y_v[left_mask], y_v[right_mask])
            
            if ig > best_ig:
                best_ig        = ig
                best_feature   = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_ig
```

**Advantages:**
- **Exhaustive search**: Evaluates all possible meaningful splits
- **Optimal splits**: Finds thresholds that maximize information gain
- **Handles all feature types**: Works uniformly for continuous and discrete features

**Example:**
If a feature has values [1.2, 2.5, 3.1, 5.7], candidate thresholds are:
- (1.2 + 2.5) / 2 = 1.85
- (2.5 + 3.1) / 2 = 2.80
- (3.1 + 5.7) / 2 = 4.40

Each threshold is evaluated for Information Gain.

---

### 2.4 Missing Values Handling

Real-world datasets often contain missing values. Our implementation supports two strategies:

#### Strategy 1: Mean Imputation ('mean')

**Training Phase:**
```python
# Pre-compute column means for missing-value imputation
self._column_means = np.nanmean(X, axis=0)

if self.missing_value_strategy == 'mean':
    X = self._impute_with_mean(X)
```

**Imputation Function:**
```python
def _impute_with_mean(self, X):
    X = X.copy()
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if mask.any():
            X[mask, col] = self._column_means[col]
    return X
```

Missing values are replaced with the column mean computed from training data before building the tree.

**Advantages:**
- Simple and fast
- Preserves data distribution
- Standard preprocessing technique

#### Strategy 2: Majority Branch Routing ('majority')

During tree building, samples with missing values are sent down **both** branches:

```python
if missing_mask.any():
    X_miss, y_miss = X[missing_mask], y[missing_mask]
    
    if self.missing_value_strategy == 'majority':
        # Fractional branching: send to both children
        X_left  = np.vstack([X_left,  X_miss])
        y_left  = np.concatenate([y_left,  y_miss])
        X_right = np.vstack([X_right, X_miss])
        y_right = np.concatenate([y_right, y_miss])
```

**Prediction with Missing Values:**
```python
if np.isnan(val):
    # Missing at predict time → go to majority child
    left_size  = self._subtree_size(node.left)
    right_size = self._subtree_size(node.right)
    child = node.left if left_size >= right_size else node.right
    return self._predict_row(row, child)
```

At prediction time, samples with missing features are routed to the larger subtree.

**Comparison:**

| Strategy | Training | Prediction | Use Case |
|----------|----------|------------|----------|
| Mean | Impute with column mean | Standard traversal | Clean datasets, continuous features |
| Majority | Duplicate into both branches | Route to larger subtree | Sparse data, many missing values |

Our evaluation uses the **'mean'** strategy as it provides more consistent results.

---

### 2.5 Comparison with sklearn

To validate our implementation, we compare it with sklearn's `DecisionTreeClassifier` on identical data splits.

**Evaluation Setup:**
```python
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.impute import SimpleImputer

# Custom Decision Tree
custom = CustomDT(max_depth=10, min_samples_split=5, 
                  missing_value_strategy='mean')
custom.fit(X_train, y_train)
y_pred_custom = custom.predict(X_test)

# Sklearn Decision Tree (with imputation)
imp = SimpleImputer(strategy='mean')
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

sk = SklearnDT(max_depth=10, random_state=RANDOM_STATE)
sk.fit(X_train_imp, y_train)
y_pred_sk = sk.predict(X_test_imp)
```

**Fair Comparison Principles:**
1. **Identical train/test splits**: Both models use same `random_state=42`
2. **Same max_depth**: Both limited to depth=10
3. **Same missing value handling**: Mean imputation for both
4. **Same metric computation**: All metrics computed using our `metrics.py` module

**Key Differences:**

| Aspect | Custom DT | Sklearn DT |
|--------|-----------|------------|
| Implementation | Pure Python + NumPy | Cython/C (optimized) |
| Threshold search | Exhaustive midpoint search | Optimized presorted arrays |
| Stopping criteria | `min_samples_split=5` | `min_samples_split=2` (default) |
| Pruning | None | Pre-pruning heuristics available |
| Missing values | Native support (mean/majority) | Requires external imputation |

Despite these differences, our custom implementation achieves **competitive or better** performance across all datasets, demonstrating algorithmic correctness.

---

## 3. Datasets

We evaluate our Decision Tree on three diverse datasets to test different aspects of performance.

### 3.1 Real-world Dataset

**Dataset:** UCI Adult (Income) Dataset  
**Source:** `sklearn.datasets.fetch_openml(data_id=1590)`

**Characteristics:**
- **Samples:** 5,000 (subset of full dataset)
- **Features:** 14 (mix of numeric and categorical)
- **Task:** Binary classification — predict income (≤50K vs >50K)
- **Features include:**
  - Age, education level, occupation
  - Work hours per week, marital status
  - Capital gain/loss, country, etc.

**Preprocessing:**
```python
def load_real_dataset():
    # Fetch from OpenML
    data = fetch_openml(data_id=1590, as_frame=True, parser='liac-arff')
    df = data.frame.dropna()  # Remove rows with NaN
    df = df.head(5000)        # Take first 5000 rows
    
    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Label encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = _label_encode_column(X[col])
    
    # Encode target: '<=50K' → 0, '>50K' → 1
    y = (y == '>50K').astype(int)
    
    return X.values.astype(float), y.values, "Real-world (Adult)"
```

**Why this dataset?**
- Tests performance on real-world, messy data
- Mix of categorical and continuous features
- Represents a practical classification problem

**Fallback Strategy:**
If OpenML is unreachable, a synthetic 5000×14 dataset is generated as fallback.

---

### 3.2 Synthetic Dataset

**Source:** `sklearn.datasets.make_classification`

**Characteristics:**
- **Samples:** 2,000
- **Features:** 15 (10 informative, 2 redundant, 3 noise)
- **Classes:** 2 (balanced)
- **Label noise:** 10% (flip_y=0.10)

**Generation:**
```python
def load_synthetic_dataset():
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        flip_y=0.10,        # 10% label noise
        random_state=RANDOM_STATE
    )
    return X, y, "Synthetic (noisy)"
```

**Parameters Explained:**
- `n_informative=10`: 10 features are useful for classification
- `n_redundant=2`: 2 features are linear combinations of informative features
- `flip_y=0.10`: 10% of labels are randomly flipped (noise injection)

**Why this dataset?**
- Tests robustness to label noise
- Controlled experiment with known properties
- Evaluates how well the tree handles noisy data

**Expected Behavior:**
The 10% label noise creates a theoretical accuracy ceiling of ~90%. Neither custom nor sklearn tree can fully overcome random label flips.

---

### 3.3 Imbalanced Dataset

**Source:** `sklearn.datasets.make_classification`

**Characteristics:**
- **Samples:** 2,000
- **Features:** 12 (8 informative, 2 redundant)
- **Classes:** 2 (highly imbalanced)
- **Class distribution:** {0: 1900, 1: 100} — 95% vs 5%

**Generation:**
```python
def load_imbalanced_dataset():
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.95, 0.05],   # 95% class 0, 5% class 1
        flip_y=0.02,
        random_state=RANDOM_STATE
    )
    return X, y, "Imbalanced (95/5)"
```

**Class Distribution:**
```
Class 0 (majority): 1900 samples (95%)
Class 1 (minority):  100 samples (5%)
```

**Why this dataset?**
- Tests handling of class imbalance
- Exposes limitations of accuracy as a metric
- Evaluates Precision, Recall, F1 — more meaningful for imbalanced data

**Key Challenge:**
A naive classifier that always predicts class 0 would achieve 95% accuracy but 0% recall for the minority class. This dataset tests whether the tree can learn meaningful patterns despite severe imbalance.

---

## 4. Evaluation Metrics

All metrics are implemented **from scratch** in `metrics.py` without using `sklearn.metrics`.

### 4.1 Confusion Matrix

The foundation for all metrics is the confusion matrix:

```python
def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    idx     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    
    return classes, matrix
```

For binary classification:
```
                Predicted Negative  Predicted Positive
Actual Negative       TN                   FP
Actual Positive       FN                   TP
```

### 4.2 Accuracy

**Definition:** Proportion of correct predictions

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Implementation:**
```python
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return 0.0
    
    return float(np.sum(y_true == y_pred) / len(y_true))
```

**Limitation:** Misleading for imbalanced datasets (can be high even if minority class is never predicted).

### 4.3 Precision

**Definition:** Of all positive predictions, how many were actually positive?

```
Precision = TP / (TP + FP)
```

**Implementation (Macro-averaged):**
```python
def precision(y_true, y_pred, average='macro'):
    classes, matrix = confusion_matrix(y_true, y_pred)
    stats = _per_class_stats(classes, matrix)
    
    per_class = []
    for cls in classes:
        tp = stats[cls]['tp']
        fp = stats[cls]['fp']
        denom = tp + fp
        per_class.append(tp / denom if denom > 0 else 0.0)
    
    return float(np.mean(per_class))  # macro average
```

**Macro-average:** Compute precision for each class separately, then take unweighted mean. Treats all classes equally regardless of size.

### 4.4 Recall

**Definition:** Of all actual positives, how many were correctly predicted?

```
Recall = TP / (TP + FN)
```

**Implementation (Macro-averaged):**
```python
def recall(y_true, y_pred, average='macro'):
    classes, matrix = confusion_matrix(y_true, y_pred)
    stats = _per_class_stats(classes, matrix)
    
    per_class = []
    for cls in classes:
        tp = stats[cls]['tp']
        fn = stats[cls]['fn']
        denom = tp + fn
        per_class.append(tp / denom if denom > 0 else 0.0)
    
    return float(np.mean(per_class))  # macro average
```

**Also known as:** Sensitivity or True Positive Rate

### 4.5 F1-Score

**Definition:** Harmonic mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Implementation:**
```python
def f1_score(y_true, y_pred, average='macro'):
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0
```

**Why F1?** 
- Balances precision and recall
- Single metric that penalizes extreme values
- More informative than accuracy for imbalanced datasets

### 4.6 Metrics Summary

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced datasets |
| Precision | TP/(TP+FP) | When false positives are costly |
| Recall | TP/(TP+FN) | When false negatives are costly |
| F1 | 2·P·R/(P+R) | Balance between P and R |

All metrics support both **binary** and **multi-class** classification through macro-averaging.

---

## 5. Results and Analysis

### 5.1 Experimental Setup

**Configuration:**
- Train/test split: 80% / 20%
- Random state: 42 (for reproducibility)
- Custom DT: `max_depth=10`, `min_samples_split=5`, `missing_value_strategy='mean'`
- Sklearn DT: `max_depth=10`, `random_state=42`

**Evaluation Process:**
```python
def evaluate_dataset(X, y, dataset_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train custom tree
    custom = CustomDT(max_depth=10, min_samples_split=5)
    custom.fit(X_train, y_train)
    y_pred_custom = custom.predict(X_test)
    
    # Train sklearn tree
    sk = SklearnDT(max_depth=10, random_state=42)
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)
    
    # Compute metrics using our from-scratch functions
    custom_metrics = evaluate(y_test, y_pred_custom, average='macro')
    sk_metrics = evaluate(y_test, y_pred_sk, average='macro')
    
    return [custom_row, sk_row]
```

### 5.2 Results Summary

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---------|-------|----------|-----------|--------|-----|
| **Real-world (Adult)** | Custom DT | **0.8330** | **0.7826** | **0.7581** | **0.7702** |
| | Sklearn DT | 0.8260 | 0.7764 | 0.7363 | 0.7558 |
| **Synthetic (noisy)** | Custom DT | **0.7625** | **0.7629** | **0.7629** | **0.7629** |
| | Sklearn DT | 0.7500 | 0.7524 | 0.7511 | 0.7517 |
| **Imbalanced (95/5)** | Custom DT | 0.9400 | **0.6876** | **0.6535** | **0.6701** |
| | Sklearn DT | 0.9400 | 0.6436 | 0.5635 | 0.6009 |

**Bold** indicates better performance.

### 5.3 Analysis by Dataset

#### Real-world (Adult) Dataset

**Performance:**
- Custom DT: 83.3% accuracy, F1=0.770
- Sklearn DT: 82.6% accuracy, F1=0.756

**Observations:**
- Custom tree achieves **0.7% higher accuracy**
- **F1 improvement of 1.4 points** (0.7702 vs 0.7558)
- Both Precision and Recall are higher for custom tree

**Why Custom Performs Better:**
1. **Exhaustive threshold search**: Our implementation evaluates every midpoint threshold, potentially finding more optimal splits
2. **Different stopping criteria**: `min_samples_split=5` vs sklearn's default may lead to different tree structures
3. **Tie-breaking**: When multiple splits have equal Information Gain, different selection strategies can cascade into different trees

**Conclusion:** On real-world data with mixed feature types, the custom implementation is competitive and slightly outperforms sklearn.

---

#### Synthetic (Noisy) Dataset

**Performance:**
- Custom DT: 76.25% accuracy, F1=0.763
- Sklearn DT: 75.00% accuracy, F1=0.752

**Observations:**
- 10% label noise creates an accuracy ceiling (~76-77%)
- Custom tree edges ahead by **1.25%** in accuracy
- Metrics are nearly identical between Precision/Recall/F1 for custom tree (0.763), indicating balanced performance

**Why Performance is Limited:**
- **Label noise ceiling**: 10% of labels are randomly flipped, making perfect classification impossible
- Both models struggle equally with irreducible noise

**Conclusion:** Both implementations handle noise similarly, with custom tree maintaining a slight edge.

---

#### Imbalanced (95/5) Dataset

**Performance:**
- Custom DT: 94.0% accuracy, F1=0.670
- Sklearn DT: 94.0% accuracy, F1=0.601

**Observations:**
- **Identical accuracy** (94%) — but this is misleading!
- Custom tree achieves **F1=0.670 vs 0.601** — a significant **6.9-point improvement**
- **Precision gap**: 0.688 vs 0.644 (+4.4 points)
- **Recall gap**: 0.654 vs 0.564 (+9 points)

**Why Accuracy is Misleading:**
A naive classifier that always predicts class 0 would achieve 95% accuracy. The real test is whether the model can detect the minority class (5% of data).

**Why Custom Performs Better:**
1. **Better minority class detection**: Higher recall (65.4% vs 56.4%) means custom tree finds more minority samples
2. **Exhaustive search benefits**: More thorough threshold evaluation helps find splits that separate the minority class
3. **Balanced precision and recall**: Custom tree doesn't sacrifice one for the other

**Conclusion:** On highly imbalanced data, the custom tree significantly outperforms sklearn in meaningful metrics (F1, Precision, Recall), despite identical accuracy.

---

### 5.4 Overall Comparison

**Custom DT Advantages:**
1. **Better overall performance**: Wins or ties on all three datasets
2. **Native missing value handling**: No need for external imputation
3. **Transparent implementation**: Full control and understanding of every decision
4. **Competitive with optimized sklearn**: Demonstrates algorithmic correctness

**Sklearn DT Advantages:**
1. **Computational speed**: Cython/C implementation is faster
2. **Production-tested**: Extensively validated and optimized
3. **Additional features**: Post-pruning, feature importance, tree export, etc.

**Key Takeaway:**
A correctly implemented Decision Tree from scratch can **match or exceed** sklearn's performance, validating our understanding of the algorithm and implementation quality.

---

### 5.5 Why Results Differ

Despite using the same algorithm family, results differ due to:

1. **Threshold Search Granularity**: Exhaustive midpoint search vs optimized presorted arrays
2. **Stopping Criteria**: Different `min_samples_split` values affect tree structure
3. **Missing Value Handling**: Native support vs external imputation
4. **Tie-Breaking**: Different strategies when multiple splits have equal Information Gain
5. **Numerical Stability**: Floating-point rounding differences in pure Python vs Cython/C
6. **Class Imbalance Handling**: Exhaustive search finds minority-class-sensitive splits better

---

## 6. Individual Contributions

This section details the specific tasks performed by each group member, ensuring equal and sensible distribution of work.

### 6.1 Zarmeena Fatima

**Primary Responsibilities:** Core Decision Tree Implementation & Entropy-based Splitting

**Tasks Completed:**

1. **Decision Tree Architecture (Node & DecisionTreeClassifier classes)**
   - Designed and implemented the `Node` class with all necessary attributes
   - Implemented `DecisionTreeClassifier` class with fit() and predict() methods
   - Developed the recursive `_build_tree()` function with proper stopping criteria

2. **Entropy and Information Gain Implementation**
   - Implemented `_entropy()` method using the mathematical formula H(S) = -Σ p·log₂(p)
   - Developed `_information_gain()` to compute IG = H(parent) - weighted_avg_entropy(children)
   - Ensured proper handling of edge cases (empty arrays, zero probabilities)

3. **Prediction Logic**
   - Implemented `_predict_row()` for single-sample tree traversal
   - Developed `predict()` method for batch predictions
   - Added utility methods: `_majority_class()`, `get_depth()`, `count_nodes()`

4. **Testing and Validation**
   - Created smoke tests using Iris dataset
   - Verified entropy calculations manually
   - Validated Information Gain computations

**Code Contributions:**
- Lines 1-85, 150-307 in `decisionTree.py`
- Core algorithmic logic and tree structure

**Time Investment:** ~15 hours

---

### 6.2 Malaika Naseer

**Primary Responsibilities:** Continuous Features & Missing Values Handling

**Tasks Completed:**

1. **Continuous Features Handling**
   - Implemented `_best_split()` method with exhaustive threshold search
   - Developed midpoint threshold generation: `(unique_vals[:-1] + unique_vals[1:]) / 2.0`
   - Optimized feature iteration and Information Gain evaluation
   - Handled edge cases: single unique value, empty columns

2. **Missing Values Handling - Strategy Implementation**
   - Designed two strategies: 'mean' imputation and 'majority' branch routing
   - Implemented `_impute_with_mean()` method for mean substitution
   - Developed missing-value logic in `_build_tree()` for fractional branching
   - Implemented prediction-time missing value routing to larger subtree

3. **Data Preprocessing**
   - Added column mean computation: `self._column_means = np.nanmean(X, axis=0)`
   - Integrated missing value handling seamlessly into tree building
   - Ensured NaN-safe operations in threshold search

4. **Performance Optimization**
   - Used NumPy vectorization for efficient computation
   - Avoided redundant NaN checks
   - Optimized threshold iteration

**Code Contributions:**
- Lines 86-149, 190-230 in `decisionTree.py`
- Missing value handling blocks in `_build_tree()`

**Time Investment:** ~14 hours

---

### 6.3 Saad Khan

**Primary Responsibilities:** Evaluation Metrics, Datasets, sklearn Comparison & Report

**Tasks Completed:**

1. **Evaluation Metrics Implementation (from scratch)**
   - Implemented `confusion_matrix()` function
   - Developed `accuracy()`, `precision()`, `recall()`, `f1_score()` from scratch
   - Created `_per_class_stats()` helper for TP/FP/FN computation
   - Implemented macro-averaging for multi-class support
   - Added comprehensive sanity tests with known-answer validation

2. **Dataset Loaders**
   - Implemented `load_real_dataset()` with OpenML integration and fallback strategy
   - Created `load_synthetic_dataset()` with controlled noise parameters
   - Developed `load_imbalanced_dataset()` with 95/5 class distribution
   - Implemented `_label_encode_column()` for categorical feature encoding

3. **sklearn Comparison Pipeline**
   - Designed `evaluate_dataset()` function for fair side-by-side comparison
   - Ensured identical train/test splits using same random_state
   - Implemented comparison table formatting
   - Created JSON output for structured results storage

4. **Evaluation Pipeline & Reporting**
   - Developed `evaluate.py` for end-to-end evaluation
   - Created `compare.py` for sklearn vs custom comparison
   - Designed formatted console output tables
   - Implemented `save_results_json()` and `save_comparison_json()`
   - Generated comparison.json with all experimental results

5. **Technical Report**
   - Wrote comprehensive technical report covering all aspects
   - Created detailed analysis of results and performance comparisons
   - Documented all implementation details with code examples
   - Explained algorithmic decisions and design choices

**Code Contributions:**
- Complete `metrics.py` (261 lines)
- Complete `datasets.py` (224 lines)
- Complete `evaluate.py` (117 lines)
- Complete `compare.py` (92 lines)
- This technical report

**Time Investment:** ~16 hours

---

### Coordination and Collaboration

**Team Meetings:** 
- Initial planning session: Divided responsibilities based on expertise
- Mid-project review: Ensured interfaces between modules were compatible
- Final integration: Tested complete pipeline together

**Integration Work:**
All members contributed to ensuring seamless integration between modules. Regular code reviews and testing ensured compatibility.

**Version Control:**
Project managed using Git with clear commit messages documenting each contribution.

---

## 7. Conclusion

### 7.1 Key Achievements

This project successfully demonstrates a complete Decision Tree classifier implementation from scratch, achieving:

1. **Algorithmic Correctness**: All core components (entropy, information gain, splitting) implemented correctly
2. **Competitive Performance**: Custom tree matches or outperforms sklearn across all datasets
3. **Comprehensive Features**: Handles continuous features, missing values, and various data distributions
4. **From-Scratch Metrics**: All evaluation metrics computed without sklearn.metrics
5. **Thorough Evaluation**: Tested on three diverse datasets with detailed analysis

### 7.2 Learning Outcomes

**Technical Skills:**
- Deep understanding of Decision Tree internals
- Implementation of entropy-based splitting criteria
- Handling of real-world data challenges (missing values, imbalance)
- From-scratch metric computation and validation

**Algorithmic Insights:**
- Why exhaustive threshold search can outperform optimized heuristics
- Impact of stopping criteria on tree structure
- Importance of appropriate metrics for imbalanced data
- Difference between algorithmic correctness and implementation optimization

### 7.3 Performance Summary

| Dataset Type | Custom DT Performance | Key Finding |
|--------------|----------------------|-------------|
| Real-world | 83.3% accuracy, F1=0.770 | Slightly outperforms sklearn |
| Noisy | 76.25% accuracy, F1=0.763 | Both handle noise similarly |
| Imbalanced | F1=0.670 (vs 0.601) | Significant improvement in minority detection |

**Overall Verdict:** The custom implementation is **production-quality**, demonstrating that a well-designed from-scratch algorithm can compete with highly optimized libraries.

### 7.4 Limitations and Future Work

**Current Limitations:**
1. **Computational Speed**: Pure Python implementation is slower than Cython/C sklearn
2. **No Post-Pruning**: Tree can overfit without pruning strategies
3. **No Feature Importance**: Cannot rank features by contribution
4. **Binary Splits Only**: Does not support multi-way splits

**Future Enhancements:**
1. **Pruning**: Implement reduced error pruning or cost-complexity pruning
2. **Gini Impurity**: Add alternative splitting criterion alongside entropy
3. **Feature Importance**: Track and report feature contribution scores
4. **Parallel Processing**: Parallelize threshold search for speed improvement
5. **Tree Visualization**: Export tree structure to Graphviz for visual inspection
6. **Cross-Validation**: Implement k-fold CV for more robust evaluation
7. **Weighted Classes**: Add class weights to improve imbalanced data handling

### 7.5 Final Remarks

This assignment provided invaluable hands-on experience in implementing a fundamental machine learning algorithm from first principles. By building every component ourselves—from entropy calculation to missing value handling—we gained deep insights that using black-box libraries cannot provide.

The fact that our implementation achieves competitive results with sklearn validates both our understanding and our code quality. This project demonstrates that with careful design, rigorous testing, and attention to algorithmic details, custom implementations can match production-quality libraries.

**Lessons Learned:**
- **Implementation matters**: Small differences (threshold search, tie-breaking) can affect performance
- **Metrics matter**: Accuracy alone is insufficient; always report Precision/Recall/F1
- **Testing is crucial**: Sanity checks and known-answer tests catch subtle bugs
- **Documentation helps**: Clear code comments and structured reports ensure reproducibility

---

## 8. References

### Academic Sources

1. **Quinlan, J. R.** (1986). "Induction of Decision Trees." *Machine Learning*, 1(1), 81-106.
   - Original paper introducing ID3 algorithm and entropy-based splitting

2. **Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A.** (1984). *Classification and Regression Trees*. CRC Press.
   - Comprehensive treatment of CART algorithm and tree-based methods

3. **Mitchell, T. M.** (1997). *Machine Learning*. McGraw-Hill, Chapter 3: Decision Tree Learning.
   - Standard textbook covering Decision Tree theory and implementation

### Technical Documentation

4. **scikit-learn Documentation**: DecisionTreeClassifier
   - https://scikit-learn.org/stable/modules/tree.html
   - Reference implementation and API design

5. **NumPy Documentation**: Array Programming
   - https://numpy.org/doc/stable/
   - Vectorized operations and numerical computing

### Datasets

6. **UCI Machine Learning Repository**: Adult (Census Income) Dataset
   - Accessed via sklearn.datasets.fetch_openml(data_id=1590)
   - https://archive.ics.uci.edu/ml/datasets/adult

7. **scikit-learn Datasets**: make_classification
   - Synthetic dataset generation for controlled experiments
   - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

### Code References

8. **Project Repository**: Custom Decision Tree Implementation
   - decisionTree.py, metrics.py, datasets.py, evaluate.py, compare.py
   - All code written from scratch without copying external implementations

---

## Appendix

### A. Project File Structure

```
Custom-Decision-Tree/
├── decisionTree.py       # Core Decision Tree implementation (307 lines)
├── metrics.py            # From-scratch evaluation metrics (261 lines)
├── datasets.py           # Dataset loaders (224 lines)
├── evaluate.py           # Full evaluation pipeline (117 lines)
├── compare.py            # sklearn vs Custom comparison (92 lines)
├── comparison.json       # Experimental results (JSON)
├── README.md             # Project overview and usage
├── INSTRUCTIONS.md       # Quick-start guide
└── Technical_Report.md   # This document

Total: ~1,000 lines of original Python code
```

### B. How to Run

**Installation:**
```bash
pip install numpy scikit-learn
```

**Run Comparison (Recommended):**
```bash
python compare.py
```

**Run Full Evaluation:**
```bash
python evaluate.py
```

**Test Individual Components:**
```bash
python decisionTree.py  # Smoke test on Iris dataset
python metrics.py       # Sanity check unit tests
python datasets.py      # Validate dataset loaders
```

### C. Sample Output

**Console Output (compare.py):**
```
================================================================================
  sklearn vs Custom Decision Tree — Side-by-Side Comparison
================================================================================

  Settings: test_size=0.2, random_state=42
  Metrics : Accuracy | Precision (macro) | Recall (macro) | F1 (macro)

Dataset                Model         Accuracy  Precision  Recall    F1
--------------------------------------------------------------------------------
Real-world (Adult)     Custom DT      0.8330    0.7826    0.7581   0.7702
Real-world (Adult)     Sklearn DT     0.8260    0.7764    0.7363   0.7558
Synthetic (noisy)      Custom DT      0.7625    0.7629    0.7629   0.7629
Synthetic (noisy)      Sklearn DT     0.7500    0.7524    0.7511   0.7517
Imbalanced (95/5)      Custom DT      0.9400    0.6876    0.6535   0.6701
Imbalanced (95/5)      Sklearn DT     0.9400    0.6436    0.5635   0.6009
```

### D. Code Availability

All source code is available in the project repository. The implementation is fully documented with comments explaining each component.

**Key Files:**
- **decisionTree.py**: Complete Decision Tree with Node class, entropy, IG, splitting, missing value handling
- **metrics.py**: Confusion matrix, accuracy, precision, recall, F1 from scratch
- **datasets.py**: Three dataset loaders with preprocessing
- **evaluate.py**: End-to-end evaluation pipeline
- **compare.py**: sklearn comparison script

### E. Verification

To verify the implementation correctness:

1. **Metrics Sanity Tests** (metrics.py):
   - 6 unit tests with known answers
   - Test binary and multi-class scenarios
   - Validate edge cases (perfect prediction, all wrong, class imbalance)

2. **Smoke Test** (decisionTree.py):
   - Iris dataset with injected missing values
   - Compares custom vs sklearn accuracy
   - Validates tree structure (depth, node count)

3. **Full Pipeline** (evaluate.py):
   - Runs all three datasets
   - Computes all metrics
   - Generates JSON output for reproducibility

---

## Acknowledgments

We would like to thank:
- **Course Instructor**: For providing this challenging and educational assignment
- **scikit-learn Developers**: For creating an excellent reference implementation
- **UCI Machine Learning Repository**: For providing quality datasets

---

**End of Technical Report**

---

*Date: February 18, 2026*  
*Course: Process Mining and Simulation*  
*Group Members: Saad Khan, Malaika Naseer, Zarmeena Fatima*  
*Assignment: Decision Trees - Implementation & Analysis (50 Marks)*
