import numpy as np

class DecisionTreeNode:
    def __init__(self, f=None, th=None, left=None, right=None, val=None):
        self.f = f
        self.th = th
        self.left = left
        self.right = right
        self.val = val

    @property
    def is_leaf(self):
        return self.left is None or self.right is None

class FastTree:
    def __init__(self, max_depth=8, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, 0)

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - float(np.sum((counts / m) ** 2))

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None
        curr_g = self._gini(y)
        best_gain = 0.0
        best_f, best_th = None, None

        for f in range(n):
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            if len(vals) > 16:
                threshs = np.quantile(vals, np.linspace(0.05, 0.95, 12))
            else:
                threshs = (vals[:-1] + vals[1:]) / 2.0

            for th in threshs:
                lm = X[:, f] <= th
                rm = ~lm
                nl, nr = int(lm.sum()), int(rm.sum())
                if nl == 0 or nr == 0:
                    continue
                gain = curr_g - (nl * self._gini(y[lm]) + nr * self._gini(y[rm])) / m
                if gain > best_gain:
                    best_gain = gain
                    best_f, best_th = f, th
        return best_f, best_th

    def _grow(self, X, y, depth):
        vals, counts = np.unique(y, return_counts=True)
        most_common = int(vals[np.argmax(counts)]) if len(vals) else 0
        if depth >= self.max_depth or len(vals) <= 1 or len(y) < self.min_samples_split:
            return DecisionTreeNode(val=most_common)

        f, th = self._best_split(X, y)
        if f is None:
            return DecisionTreeNode(val=most_common)

        lm = X[:, f] <= th
        return DecisionTreeNode(f=f, th=th,
                                left=self._grow(X[lm], y[lm], depth + 1),
                                right=self._grow(X[~lm], y[~lm], depth + 1),
                                val=None)

    def predict(self, X):
        out = np.zeros(len(X), dtype=int)
        def eval_node(node, idx):
            if len(idx) == 0: return
            if node.is_leaf:
                out[idx] = node.val
                return
            cond = X[idx, node.f] <= node.th
            eval_node(node.left, idx[cond])
            eval_node(node.right, idx[~cond])
        eval_node(self.root, np.arange(len(X)))
        return out

class CANForest:
    def __init__(self, n_estimators=8, max_depth=8, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.trees = []
        for i in range(self.n_estimators):
            sample_size = min(n, 20000)
            idx = rng.choice(n, size=sample_size, replace=True)
            tree = FastTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        if not self.trees:
            return np.zeros(len(X), dtype=int)
        preds = np.column_stack([t.predict(X) for t in self.trees])
        out = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            vals, counts = np.unique(preds[i], return_counts=True)
            out[i] = vals[np.argmax(counts)]
        return out
