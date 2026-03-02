# svm.py  (UPDATED)
import numpy as np


class KernelSVM:
    """
    Kernel SVM (SMO-style) updated for your gait sliding-window dataset.

    Key updates vs your old version:
    1) max_samples: optional subsampling to avoid n×n kernel matrix explosion
    2) seeded randomness: reproducible choice of j
    3) support-vector compression kept (as you wanted)
    """

    def __init__(self, C=1.0, tol=1e-3, max_iter=1000, seed=None, max_samples=None):
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        # NEW: cap the number of training samples to keep K (n×n) manageable
        self.max_samples = None if max_samples is None else int(max_samples)

        # NEW: reproducible RNG (optional)
        self.rng = np.random.default_rng(seed) if seed is not None else None

        # Learned params
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
        self.b = 0.0

    def kernel(self, X1: np.ndarray, X2: np.ndarray | None = None):
        """Override in subclasses."""
        raise NotImplementedError

    def _pick_j(self, n: int, i: int) -> int:
        """Pick j != i (seeded if rng is set)."""
        if self.rng is None:
            j = i
            while j == i:
                j = np.random.randint(n)
            return int(j)
        j = i
        while j == i:
            j = int(self.rng.integers(0, n))
        return j

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1).astype(float)

        # Expect {-1, +1}
        assert set(np.unique(y)) == {-1.0, 1.0}, "Labels must be binary (-1 and 1)."

        n = X.shape[0]

        # NEW: subsample BEFORE building the kernel matrix
        if self.max_samples is not None and n > self.max_samples:
            if self.rng is None:
                idx = np.random.choice(n, self.max_samples, replace=False)
            else:
                idx = self.rng.choice(n, self.max_samples, replace=False)

            X = X[idx]
            y = y[idx]
            n = X.shape[0]

        # Full kernel matrix [n, n] (this is why we cap n)
        K = self.kernel(X)

        alphas = np.zeros(n, dtype=float)
        b = 0.0

        # Error cache: E_i = f(x_i) - y_i
        # Initially f(x)=0 => E = -y
        E = -y.copy()

        for _ in range(self.max_iter):
            num_changed = 0

            for i in range(n):
                # KKT violation check
                violates = (
                    (y[i] * E[i] < -self.tol and alphas[i] < self.C)
                    or (y[i] * E[i] > self.tol and alphas[i] > 0.0)
                )
                if not violates:
                    continue

                j = self._pick_j(n, i)

                # Compute L and H bounds
                if y[i] != y[j]:
                    L = max(0.0, alphas[j] - alphas[i])
                    H = min(self.C, self.C + alphas[j] - alphas[i])
                else:
                    L = max(0.0, alphas[i] + alphas[j] - self.C)
                    H = min(self.C, alphas[i] + alphas[j])

                if L >= H:
                    continue

                # eta = 2*Kij - Kii - Kjj
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0.0:
                    continue

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                b_old = b

                # Update alpha_j
                alphas[j] -= y[j] * (E[i] - E[j]) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                # Update b (b1 / b2)
                b1 = (
                    b
                    - E[i]
                    - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                    - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                )
                b2 = (
                    b
                    - E[j]
                    - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                    - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                )

                if 0.0 < alphas[i] < self.C:
                    b = b1
                elif 0.0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                # Incremental error update
                delta_i = (alphas[i] - alpha_i_old) * y[i]
                delta_j = (alphas[j] - alpha_j_old) * y[j]
                E += delta_i * K[i] + delta_j * K[j] + (b - b_old)

                num_changed += 1

            # Stop if no alpha updated this full pass
            if num_changed == 0:
                break

        # Support-vector compression
        sv_mask = alphas > 1e-5
        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.b = float(b)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.alphas is not None
        assert self.support_vectors is not None
        X = np.asarray(X, dtype=float)
        K = self.kernel(X, self.support_vectors)
        return K @ (self.alphas * self.support_labels) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))


class RBFSVM(KernelSVM):
    """
    RBF kernel SVM with gamma locked during fit (so it doesn't accidentally change later).
    """

    def __init__(self, gamma=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        # NEW: set gamma once using training dimension if not provided
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        return super().fit(X, y)

    def kernel(self, X1: np.ndarray, X2: np.ndarray | None = None):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)

        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist2 = sq1 + sq2 - 2.0 * (X1 @ X2.T)
        return np.exp(-float(self.gamma) * dist2)


class PolySVM(KernelSVM):
    def __init__(self, degree=2, coef0=1.0, **kwargs):
        super().__init__(**kwargs)
        self.degree = int(degree)
        self.coef0 = float(coef0)

    def kernel(self, X1: np.ndarray, X2: np.ndarray | None = None):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        return (X1 @ X2.T + self.coef0) ** self.degree