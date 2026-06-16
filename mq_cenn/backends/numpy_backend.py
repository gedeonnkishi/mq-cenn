from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NumpyBackend:
    """
    Reference backend for MQ-CeNN.

    This backend is always available. It is CPU-based, portable and used as
    the correctness reference for future C++ and CUDA backends.
    """

    name: str = "numpy"
    supports_cuda: bool = False
    supports_autograd: bool = False

    def as_float64(self, x, *, name: str = "array") -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)

        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values.")

        return arr

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def mean(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def var(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.var(x, axis=axis, keepdims=keepdims)

    def std(
        self,
        x: np.ndarray,
        axis=None,
        keepdims: bool = False,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return np.std(x, axis=axis, keepdims=keepdims) + float(eps)

    def column_stack(self, arrays) -> np.ndarray:
        return np.column_stack(arrays)

    def ridge_solve(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Solve the ridge system:

            beta = argmin ||Z beta - y||² + alpha ||beta||²

        using the normal equation with a least-squares fallback.
        """
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        alpha = float(alpha)

        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array.")

        if Z.shape[0] != y.shape[0]:
            raise ValueError("Z and y length mismatch.")

        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        A = Z.T @ Z
        A.flat[:: A.shape[0] + 1] += alpha

        rhs = Z.T @ y

        try:
            return np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, rhs, rcond=None)[0]


_NUMPY_BACKEND: Optional[NumpyBackend] = None


def get_numpy_backend() -> NumpyBackend:
    global _NUMPY_BACKEND

    if _NUMPY_BACKEND is None:
        _NUMPY_BACKEND = NumpyBackend()

    return _NUMPY_BACKEND


__all__ = [
    "NumpyBackend",
    "get_numpy_backend",
]
