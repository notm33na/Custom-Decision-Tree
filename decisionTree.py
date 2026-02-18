
import numpy as np
from collections import Counter
#  Node
class Node:
    """A single node in the Decision Tree."""

    def __init__(self):
        self.feature_index = None   # Index of feature to split on
        self.threshold     = None   # Threshold for continuous features
        self.left          = None   # Left child  (feature <= threshold)
        self.right         = None   # Right child (feature >  threshold)
        self.is_leaf       = False
        self.label         = None   # Predicted class (leaf nodes only)

#  Decision Tree
class DecisionTreeClassifier:
    """
    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None = grow until pure or min_samples_split.
    min_samples_split : int
        Minimum samples required to attempt a split.
    missing_value_strategy : str
        'mean'     – impute missing values with column mean before training.
        'majority' – send missing-value samples down both branches (fractional).
        Default: 'mean'
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 missing_value_strategy='mean'):
        self.max_depth               = max_depth
        self.min_samples_split       = min_samples_split
        self.missing_value_strategy  = missing_value_strategy
        self.root                    = None
        self._column_means           = None   # Stored for 'mean' strategy

    # ── Public API ────────────────────────────

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        # Pre-compute column means for missing-value imputation
        self._column_means = np.nanmean(X, axis=0)

        if self.missing_value_strategy == 'mean':
            X = self._impute_with_mean(X)

        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        if self.missing_value_strategy == 'mean':
            X = self._impute_with_mean(X)
        return np.array([self._predict_row(row, self.root) for row in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    # Missing value helpers 

    def _impute_with_mean(self, X):
        """Replace NaN with per-column mean (computed from training data)."""
        X = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                X[mask, col] = self._column_means[col]
        return X

    # Entropy andd Information Gainn

    @staticmethod
    def _entropy(y):
        """
        H(S) = - Σ p_i * log2(p_i)
        Returns 0 for empty arrays.
        """
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, y, y_left, y_right):
        """
        IG = H(parent) - weighted_avg_entropy(children)
        """
        n       = len(y)
        n_l     = len(y_left)
        n_r     = len(y_right)

        if n_l == 0 or n_r == 0:
            return 0.0

        parent_entropy = self._entropy(y)
        child_entropy  = (n_l / n) * self._entropy(y_left) + \
                         (n_r / n) * self._entropy(y_right)
        return parent_entropy - child_entropy

    # funcrion for Best Split

    def _best_split(self, X, y):
        """
        Search every feature and every unique mid-point threshold.
        Returns (best_feature_index, best_threshold, best_ig).
        """
        n_features      = X.shape[1]
        best_ig         = -1
        best_feature    = None
        best_threshold  = None

        for feature_idx in range(n_features):
            col     = X[:, feature_idx]

            # Ignore NaN rows when scanning the thresholds
            valid   = ~np.isnan(col)
            col_v   = col[valid]
            y_v     = y[valid]

            if len(col_v) == 0:
                continue

            unique_vals = np.unique(col_v)
            if len(unique_vals) == 1:
                continue

            # Candidate thresholds: midpoints between consecutive sorted values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask  = col_v <= threshold
                right_mask = ~left_mask

                ig = self._information_gain(y_v,
                                            y_v[left_mask],
                                            y_v[right_mask])
                if ig > best_ig:
                    best_ig        = ig
                    best_feature   = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_ig

    # ── Tree Building starts frm here

    def _build_tree(self, X, y, depth):
        node = Node()

        # Stopping criteria defined here
        is_pure        = len(np.unique(y)) == 1
        too_small      = len(y) < self.min_samples_split
        max_depth_hit  = (self.max_depth is not None) and (depth >= self.max_depth)

        if is_pure or too_small or max_depth_hit:
            node.is_leaf = True
            node.label   = self._majority_class(y)
            return node

        # Find best split
        feature_idx, threshold, ig = self._best_split(X, y)

        if feature_idx is None or ig <= 0:
            node.is_leaf = True
            node.label   = self._majority_class(y)
            return node

        node.feature_index = feature_idx
        node.threshold     = threshold

        # Partition samples
        col        = X[:, feature_idx]
        known_mask = ~np.isnan(col)
        missing_mask = np.isnan(col)

        left_mask  = known_mask & (col <= threshold)
        right_mask = known_mask & (col >  threshold)

        X_left, y_left   = X[left_mask],  y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        #Handle samples with missing feature value
        if missing_mask.any():
            X_miss, y_miss = X[missing_mask], y[missing_mask]

            if self.missing_value_strategy == 'majority':
                # Fractional branching: send to both children (duplicated)
                X_left  = np.vstack([X_left,  X_miss])
                y_left  = np.concatenate([y_left,  y_miss])
                X_right = np.vstack([X_right, X_miss])
                y_right = np.concatenate([y_right, y_miss])
            else:
                # 'mean' strategy: shouldn't reach here normally,
                # but fall back to majority side
                n_l = len(X_left)
                n_r = len(X_right)
                if n_l >= n_r:
                    X_left  = np.vstack([X_left,  X_miss])
                    y_left  = np.concatenate([y_left,  y_miss])
                else:
                    X_right = np.vstack([X_right, X_miss])
                    y_right = np.concatenate([y_right, y_miss])

        if len(X_left) == 0 or len(X_right) == 0:
            node.is_leaf = True
            node.label   = self._majority_class(y)
            return node

        node.left  = self._build_tree(X_left,  y_left,  depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    # Prediction fnctn

    def _predict_row(self, row, node):
        if node.is_leaf:
            return node.label

        val = row[node.feature_index]

        if np.isnan(val):
            # Missing at predict time → go majority child
            left_size  = self._subtree_size(node.left)
            right_size = self._subtree_size(node.right)
            child = node.left if left_size >= right_size else node.right
            return self._predict_row(row, child)

        if val <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    # Utilities

    @staticmethod
    def _majority_class(y):
        if len(y) == 0:
            return 0
        return Counter(y.tolist()).most_common(1)[0][0]

    def _subtree_size(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._subtree_size(node.left) + self._subtree_size(node.right)

    def get_depth(self):
        """Return the actual depth of the fitted tree."""
        return self._depth(self.root)

    def _depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def count_nodes(self):
        return self._count(self.root)

    def _count(self, node):
        if node is None:
            return 0
        return 1 + self._count(node.left) + self._count(node.right)



#  Quick smoke test
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X, y = data.data, data.target

    # Inject some missing values to verify handling
    rng = np.random.default_rng(42)
    mask = rng.random(X.shape) < 0.05
    X_missing = X.copy().astype(float)
    X_missing[mask] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=5, missing_value_strategy='mean')
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Custom DT Accuracy  : {acc:.4f}")
    print(f"Tree Depth          : {clf.get_depth()}")
    print(f"Total Nodes         : {clf.count_nodes()}")

    # Compare with sklearn
    from sklearn.tree import DecisionTreeClassifier as SklearnDT
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy='mean')
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp  = imp.transform(X_test)

    sk_clf = SklearnDT(max_depth=5, random_state=42)
    sk_clf.fit(X_train_imp, y_train)
    sk_acc = sk_clf.score(X_test_imp, y_test)
    print(f"Sklearn DT Accuracy : {sk_acc:.4f}")