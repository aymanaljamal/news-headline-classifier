import numpy as np


class LogisticRegressionOVR:
    def __init__(self, max_iterations=100, learning_rate=0.05, batch_size=256, verbose=False):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

        self.classes = None
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _train_binary_minibatch(self, X, y, rng):
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=float)
        b = 0.0

        for epoch in range(1, self.max_iterations + 1):
            idx = rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                m = len(batch_idx)
                if m == 0:
                    continue

                Xb = X[batch_idx]
                yb = y[batch_idx]


                z = np.dot(Xb, w) + b
                pred = self._sigmoid(z)
                err = pred - yb


                grad_w = np.dot(Xb.T, err) / m
                grad_b = err.mean()

                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b

            if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.max_iterations):
                print(f"  [LR] epoch {epoch}/{self.max_iterations}")

        return w, b

    def fit(self, X, y, seed=42):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.array(sorted(set(y)))
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.weights = np.zeros((n_classes, n_features), dtype=float)
        self.bias = np.zeros(n_classes, dtype=float)

        rng = np.random.default_rng(seed)

        for i, cls in enumerate(self.classes):
            if self.verbose:
                print(f"Training OVR: class {cls} vs rest ...")

            binary = (y == cls).astype(float)
            w, b = self._train_binary_minibatch(X, binary, rng)
            self.weights[i] = w
            self.bias[i] = b

    def predict(self, X):
        X = np.asarray(X, dtype=float)


        scores = np.dot(X, self.weights.T) + self.bias

        best_idx = np.argmax(scores, axis=1)
        return self.classes[best_idx].tolist()
