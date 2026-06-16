from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from mq_cenn.utils.validation import _as_2d_float64


KernelName = Literal[
    "gaussian",
    "matern32",
    "laplacian",
    "periodic",
    "polynomial",
]


@dataclass(frozen=True)
class KernelSpec:
    """
    Specification of one kernel-inspired expert family.

    The implementation remains classical and claim-bounded:
    these kernels are used as random-feature / random-kitchen-sink mappings,
    not as quantum circuits.
    """

    name: KernelName
    gamma: float = 1.0
    period: float = 24.0
    degree: int = 2


DEFAULT_KERNEL_SPECS: Tuple[KernelSpec, ...] = (
    KernelSpec("gaussian", gamma=1.0),
    KernelSpec("matern32", gamma=0.8),
    KernelSpec("laplacian", gamma=0.5),
    KernelSpec("periodic", gamma=1.0, period=24.0),
    KernelSpec("polynomial", gamma=1.0, degree=2),
)


class SpectralFeatureProjector:
    """
    Random feature projector for kernel-inspired lifting.

    Supported families
    ------------------
    gaussian:
        Random Fourier features with Gaussian spectral measure.

    matern32:
        Heavy-tailed spectral proxy inspired by Matérn-type kernels.

    laplacian:
        Cauchy spectral proxy with clipping for numerical stability.

    periodic:
        Harmonic random-kitchen-sink proxy.

    polynomial:
        Bounded polynomial random projection proxy.

    Notes
    -----
    This class does not implement quantum computation. It implements
    classical random-feature mappings used as quantum-inspired proxies.
    """

    SUPPORTED: Tuple[str, ...] = (
        "gaussian",
        "matern32",
        "laplacian",
        "periodic",
        "polynomial",
    )

    def __init__(
        self,
        spec: KernelSpec,
        n_features: int = 512,
        random_state: int = 42,
    ) -> None:
        if spec.name not in self.SUPPORTED:
            raise ValueError(f"Unsupported kernel family: {spec.name!r}")

        if int(n_features) < 4:
            raise ValueError("n_features must be >= 4.")

        self.spec = spec
        self.n_features = int(n_features)
        self.random_state = int(random_state)

        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None
        self.scale_: float = float(np.sqrt(2.0 / self.n_features))

    def fit(self, n_input_features: int) -> "SpectralFeatureProjector":
        """
        Initialize random projection parameters.

        Parameters
        ----------
        n_input_features:
            Number of input features.
        """
        d = int(n_input_features)

        if d <= 0:
            raise ValueError("n_input_features must be positive.")

        rng = np.random.default_rng(self.random_state)
        gamma = max(float(self.spec.gamma), 1e-12)

        if self.spec.name == "gaussian":
            self.W_ = rng.normal(
                0.0,
                np.sqrt(2.0 * gamma),
                size=(d, self.n_features),
            )
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "matern32":
            nu = 3.0
            z = rng.normal(0.0, 1.0, size=(d, self.n_features))
            v = rng.chisquare(nu, size=(1, self.n_features)) / nu
            self.W_ = np.sqrt(3.0 * gamma) * z / np.sqrt(v + 1e-12)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "laplacian":
            u = rng.uniform(1e-6, 1.0 - 1e-6, size=(d, self.n_features))
            self.W_ = gamma * np.tan(np.pi * (u - 0.5))
            self.W_ = np.clip(self.W_, -50.0, 50.0)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "periodic":
            period = max(float(self.spec.period), 1e-6)
            base = 2.0 * np.pi / period
            max_harmonic = max(2, self.n_features // 2)

            harmonics = rng.integers(
                1,
                max_harmonic,
                size=(d, self.n_features),
            )
            signs = rng.choice([-1.0, 1.0], size=(d, self.n_features))

            self.W_ = signs * harmonics * base * np.sqrt(gamma)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "polynomial":
            self.W_ = rng.normal(
                0.0,
                np.sqrt(gamma),
                size=(d, self.n_features),
            )
            self.b_ = rng.normal(0.0, 1.0, size=self.n_features)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data into random-feature space.
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Projector must be fitted before transform().")

        X = _as_2d_float64(X, name="X")

        if X.shape[1] != self.W_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but projector was fitted "
                f"with {self.W_.shape[0]} features."
            )

        z = X @ self.W_ + self.b_

        if self.spec.name == "polynomial":
            degree = max(1, int(self.spec.degree))
            z = np.tanh(z)

            if degree > 1:
                z = np.sign(z) * (np.abs(z) ** degree)

            return z / np.sqrt(self.n_features)

        return self.scale_ * np.cos(z)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the projector from X and return transformed features.
        """
        X = _as_2d_float64(X, name="X")
        return self.fit(X.shape[1]).transform(X)


__all__ = [
    "KernelName",
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "SpectralFeatureProjector",
]
