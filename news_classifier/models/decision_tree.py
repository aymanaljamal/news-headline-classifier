from collections import Counter
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, max_thresholds=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.root = None
        self.default_class = None

    def _gini(self, y):
        if y is None or len(y) == 0:
            return 0.0
        y = np.asarray(y)
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p * p)


    def _majority(self, y):
        if y is None or len(y) == 0:
            return self.default_class

        return Counter(y).most_common(1)[0][0]

    def _split(self, X, y, j, thr):
        mask = X[:, j] <= thr
        lx = X[mask]
        ly = y[mask]
        rx = X[~mask]
        ry = y[~mask]
        return lx, ly, rx, ry

    def _best_split(self, X, y):
        if X is None or len(X) == 0:
            return None, None

        n_samples, n_features = X.shape
        best_feat, best_thr, best_gini = None, None, float("inf")

        for j in range(n_features):
            col = X[:, j]
            values = np.unique(col)
            if values.size < 2:
                continue


            if values.size > self.max_thresholds:
                values = np.quantile(col, np.linspace(0, 1, self.max_thresholds))
                values = np.unique(values)
                if values.size < 2:
                    continue

            # try midpoints
            for k in range(values.size - 1):
                thr = (values[k] + values[k + 1]) / 2.0

                mask = col <= thr
                ly = y[mask]
                ry = y[~mask]

                if ly.size == 0 or ry.size == 0:
                    continue

                g = (ly.size / n_samples) * self._gini(ly) + (ry.size / n_samples) * self._gini(ry)
                if g < best_gini:
                    best_feat, best_thr, best_gini = j, thr, g

        return best_feat, best_thr

    def _build(self, X, y, depth):
        if y is None or len(y) == 0:
            return Node(value=self.default_class)

        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return Node(value=self._majority(y))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return Node(value=self._majority(y))

        lx, ly, rx, ry = self._split(X, y, feat, thr)
        if ly.size == 0 or ry.size == 0:
            return Node(value=self._majority(y))

        left_node = self._build(lx, ly, depth + 1)
        right_node = self._build(rx, ry, depth + 1)
        return Node(feature=feat, threshold=thr, left=left_node, right=right_node)

    def fit(self, X, y):
        if y is None or len(y) == 0:
            raise ValueError("DecisionTree.fit: y is empty. Check your data loading/splitting.")

        X = np.asarray(X)
        y = np.asarray(y)

        self.default_class = Counter(y).most_common(1)[0][0]
        self.root = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if node is None:
            return self.default_class

        while not node.is_leaf():
            if node.feature is None or node.threshold is None:
                return self.default_class

            node = node.left if x[node.feature] <= node.threshold else node.right

            if node is None:
                return self.default_class

        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return [self._predict_one(x, self.root) for x in X]
