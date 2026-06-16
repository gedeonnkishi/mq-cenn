from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from mq_cenn.core.kernels import DEFAULT_KERNEL_SPECS, KernelSpec, SpectralFeatureProjector
from mq_cenn.utils.validation import _as_1d_float64, _as_2d_float64


class KernelRidgeExpert:
    """
    One random-feature ridge expert.

    Each expert owns its own spectral/random-kitchen-sink projector and
    a closed-form ridge regression head.
    """

    def __init__(
        self,
        spec: KernelSpec,
        n_features: int,
        alpha: float,
        random_state: int,
    ) -> None:
        if int(n_features) < 4:
            raise ValueError("n_features must be >= 4.")

        if float(alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

        self.spec = spec
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

        self.projector_: Optional[SpectralFeatureProjector] = None
        self.beta_: Optional[np.ndarray] = None
        self.train_residual_std_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeExpert":
        X = _as_2d_float64(X, name="X")
        y = _as_1d_float64(y, name="y")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")

        self.projector_ = SpectralFeatureProjector(
            spec=self.spec,
            n_features=self.n_features,
            random_state=self.random_state,
        ).fit(X.shape[1])

        Z = self.projector_.transform(X)

        A = Z.T @ Z
        A.flat[:: A.shape[0] + 1] += self.alpha

        rhs = Z.T @ y

        try:
            self.beta_ = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(A, rhs, rcond=None)[0]

        residuals = y - (Z @ self.beta_)
        self.train_residual_std_ = float(np.std(residuals) + 1e-8)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.projector_ is None or self.beta_ is None:
            raise RuntimeError("Expert must be fitted before predict().")

        X = _as_2d_float64(X, name="X")
        return self.projector_.transform(X) @ self.beta_


class MultiKernelExpertPool:
    """
    Heterogeneous pool of kernel-inspired experts.

    The pool combines several kernel families and several random projections
    per family. This creates a multi-hypothesis representation before the
    neural bridge and signed gate.
    """

    def __init__(
        self,
        kernel_specs: Sequence[KernelSpec] = DEFAULT_KERNEL_SPECS,
        n_experts_per_kernel: int = 2,
        n_features_per_expert: int = 512,
        alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        n_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self.kernel_specs = tuple(kernel_specs)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.n_features_per_expert = int(n_features_per_expert)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

        if not self.kernel_specs:
            raise ValueError("kernel_specs must contain at least one KernelSpec.")

        if self.n_experts_per_kernel < 1:
            raise ValueError("n_experts_per_kernel must be >= 1.")

        if self.n_features_per_expert < 4:
            raise ValueError("n_features_per_expert must be >= 4.")

        if not self.alpha_grid:
            raise ValueError("alpha_grid must not be empty.")

        if any(alpha <= 0.0 for alpha in self.alpha_grid):
            raise ValueError("All alpha values must be positive.")

        self.experts_: List[KernelRidgeExpert] = []
        self.best_alpha_: Dict[str, float] = {}

    @property
    def n_experts_(self) -> int:
        return len(self.experts_)

    def _select_alpha(self, X: np.ndarray, y: np.ndarray, spec: KernelSpec) -> float:
        """
        Select ridge alpha using chronological TimeSeriesSplit.

        For very small datasets, a middle alpha is returned to avoid unstable
        validation folds.
        """
        if X.shape[0] < max(40, self.n_splits + 5):
            return self.alpha_grid[len(self.alpha_grid) // 2]

        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        best_alpha = self.alpha_grid[0]
        best_loss = np.inf

        for alpha in self.alpha_grid:
            losses = []

            for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X)):
                expert = KernelRidgeExpert(
                    spec=spec,
                    n_features=self.n_features_per_expert,
                    alpha=alpha,
                    random_state=self.random_state + 1009 * fold_id,
                )
                expert.fit(X[train_idx], y[train_idx])

                pred = expert.predict(X[valid_idx])
                loss = float(np.mean(np.abs(pred - y[valid_idx])))
                losses.append(loss)

            mean_loss = float(np.mean(losses))

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_alpha = float(alpha)

        return best_alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiKernelExpertPool":
        X = _as_2d_float64(X, name="X")
        y = _as_1d_float64(y, name="y")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")

        self.experts_ = []
        self.best_alpha_ = {}

        seed_counter = 0

        for spec in self.kernel_specs:
            alpha = self._select_alpha(X, y, spec)
            self.best_alpha_[spec.name] = alpha

            for _ in range(self.n_experts_per_kernel):
                expert = KernelRidgeExpert(
                    spec=spec,
                    n_features=self.n_features_per_expert,
                    alpha=alpha,
                    random_state=self.random_state + 9973 * seed_counter,
                )

                expert.fit(X, y)
                self.experts_.append(expert)
                seed_counter += 1

        return self

    def predict_pool(self, X: np.ndarray) -> np.ndarray:
        """
        Return one prediction column per expert.

        Shape
        -----
        (n_samples, n_experts)
        """
        if not self.experts_:
            raise RuntimeError("Expert pool must be fitted before predict_pool().")

        X = _as_2d_float64(X, name="X")

        return np.column_stack([expert.predict(X) for expert in self.experts_])

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Return the arithmetic mean of the expert pool predictions.
        """
        return self.predict_pool(X).mean(axis=1)


__all__ = [
    "KernelRidgeExpert",
    "MultiKernelExpertPool",
]
