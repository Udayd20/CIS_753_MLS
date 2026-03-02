import numpy as np
class KernelSVM_RBF:

    def __init__(self, C=1.0, gamma=1.0, tol=1e-3, max_passes=8, seed=0):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.rng = np.random.default_rng(seed)

        self.X = None
        self.y = None
        self.alphas = None
        self.b = 0.0
        self.K = None

    def _to_pm_one(self, y):
        y = np.asarray(y).reshape(-1)
        return np.where(y <= 0, -1, 1).astype(float)

    def rbf_kernel(self, X1, X2):
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        dist2 = sq1 - 2 * (X1 @ X2.T) + sq2
        return np.exp(-self.gamma * dist2)

    def _decision_train_i(self, i):
        return float((self.alphas * self.y) @ self.K[:, i] + self.b)

    def fit(self, X, y, verbose=False):
        X = np.asarray(X, dtype=float)
        y = self._to_pm_one(y)
        n = X.shape[0]

        self.X = X
        self.y = y
        self.alphas = np.zeros(n, dtype=float)
        self.b = 0.0

        self.K = self.rbf_kernel(X, X)

        passes = 0
        iters = 0

        while passes < self.max_passes:
            num_changed = 0
            iters += 1

            for i in range(n):
                Ei = self._decision_train_i(i) - self.y[i]

                if ((self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (self.y[i] * Ei >  self.tol and self.alphas[i] > 0)):

                    j = i
                    while j == i:
                        j = self.rng.integers(0, n)

                    Ej = self._decision_train_i(j) - self.y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    if self.y[i] != self.y[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    b1 = (self.b - Ei
                          - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i]
                          - self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j])

                    b2 = (self.b - Ej
                          - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j]
                          - self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                    num_changed += 1

            if verbose:
                sv = np.sum(self.alphas > 1e-8)
                print(f"SMO iter {iters:3d}: changed={num_changed:3d}, passes={passes}, support_vectors={sv}")

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K_test = self.rbf_kernel(X, self.X)
        return (K_test @ (self.alphas * self.y)) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def hinge_loss(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._to_pm_one(y)
        margins = y * self.decision_function(X)
        return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def accuracy_score_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def train_test_split_np(X, y, test_size=0.2, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_toy_circle(n_samples=300, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, 2))
    y = np.where(X[:, 0]**2 + X[:, 1]**2 > 1.0, 1, -1)
    return X, y


if __name__ == "__main__":
    X, y = make_toy_circle(n_samples=300, seed=1)
    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, seed=42)

    rbf = KernelSVM_RBF(C=1.0, gamma=1.0, tol=1e-3, max_passes=8, seed=42)
    rbf.fit(X_train, y_train, verbose=True)

    y_pred_train = rbf.predict(X_train)
    y_pred_test = rbf.predict(X_test)

    print("\nLearned parameters:")
    print("b =", rbf.b)
    print("Number of support vectors:", int(np.sum(rbf.alphas > 1e-8)))

    print("\nAccuracy:")
    print(f"Train accuracy: {accuracy_score_np(y_train, y_pred_train):.3f}")
    print(f"Test  accuracy: {accuracy_score_np(y_test, y_pred_test):.3f}")

    print("\nHinge loss:")
    print("Train hinge loss:", rbf.hinge_loss(X_train, y_train))
    print("Test  hinge loss:", rbf.hinge_loss(X_test, y_test))
