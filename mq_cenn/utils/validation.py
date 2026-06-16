from __future__ import annotations

from typing import Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


def as_float64(x: ArrayLike, *, name: str = "array") -> np.ndarray:
    """
    Convert input data to a finite NumPy float64 array.

    Parameters
    ----------
    x:
        Input array-like object.
    name:
        Human-readable variable name used in error messages.

    Returns
    -------
    np.ndarray
        Converted float64 array.

    Raises
    ------
    ValueError
        If the array contains NaN or infinite values.
    """
    arr = np.asarray(x, dtype=np.float64)

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def as_2d_float64(x: ArrayLike, *, name: str = "X") -> np.ndarray:
    """
    Convert input data to a finite 2D NumPy float64 array.

    Parameters
    ----------
    x:
        Input feature matrix.
    name:
        Human-readable variable name used in error messages.

    Returns
    -------
    np.ndarray
        2D float64 array.

    Raises
    ------
    ValueError
        If the input is not 2D.
    """
    arr = as_float64(x, name=name)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")

    return arr


def as_1d_float64(y: ArrayLike, *, name: str = "y") -> np.ndarray:
    """
    Convert target data to a finite 1D NumPy float64 array.

    Parameters
    ----------
    y:
        Target values.
    name:
        Human-readable variable name used in error messages.

    Returns
    -------
    np.ndarray
        Flattened 1D float64 array.
    """
    return as_float64(y, name=name).reshape(-1)


def safe_std(
    x: np.ndarray,
    axis=None,
    keepdims: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a numerically stable standard deviation.

    A small epsilon is added to avoid division by zero during normalization.

    Parameters
    ----------
    x:
        Input array.
    axis:
        Axis or axes along which the standard deviation is computed.
    keepdims:
        Whether to keep reduced dimensions.
    eps:
        Small positive value added to the standard deviation.

    Returns
    -------
    np.ndarray
        Standard deviation plus epsilon.
    """
    return np.std(x, axis=axis, keepdims=keepdims) + float(eps)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

_as_float64 = as_float64
_as_2d_float64 = as_2d_float64
_as_1d_float64 = as_1d_float64
_safe_std = safe_std


__all__ = [
    "ArrayLike",
    "as_float64",
    "as_2d_float64",
    "as_1d_float64",
    "safe_std",
    "_as_float64",
    "_as_2d_float64",
    "_as_1d_float64",
    "_safe_std",
]
