"""
Windowing utilities for time-series forecasting.

The main goal is to build strict forecasting samples:

    X_t = values[t-lookback : t]
    y_t = target[t+horizon-1]

No future value is included in X_t.

The functions support both univariate and multivariate series. MQ-CeNN's
current regressor expects a two-dimensional matrix, so windows can be flattened
into shape:

    (n_samples, lookback * n_features)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class WindowMetadata:
    """Metadata returned when `return_metadata=True`."""

    lookback: int
    horizon: int
    n_features: int
    flattened: bool
    last_value_index: int
    sample_start_indices: np.ndarray
    target_indices: np.ndarray


def _as_2d_values(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(
            "values must be a 1D or 2D array-like object. "
            f"Received shape {arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains NaN or infinite values.")

    return arr


def _as_1d_target(target_values) -> np.ndarray:
    arr = np.asarray(target_values, dtype=np.float64)

    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)

    if arr.ndim != 1:
        raise ValueError(
            "target_values must be a 1D array-like object. "
            f"Received shape {arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("target_values contains NaN or infinite values.")

    return arr


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D windows into a 2D matrix.

    Parameters
    ----------
    X:
        Array of shape `(n_samples, lookback, n_features)`.

    Returns
    -------
    np.ndarray
        Array of shape `(n_samples, lookback * n_features)`.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(f"Expected a 3D window tensor, received shape {X.shape}.")

    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def make_supervised_windows(
    values,
    target_values=None,
    *,
    target_index: int = 0,
    lookback: int = 24,
    horizon: int = 1,
    flatten: bool = True,
    max_samples: Optional[int] = None,
    return_metadata: bool = False,
):
    """
    Build supervised forecasting windows.

    Parameters
    ----------
    values:
        Univariate or multivariate sequence of shape `(n_timesteps,)` or
        `(n_timesteps, n_features)`.

    target_values:
        Optional explicit target series of shape `(n_timesteps,)`.
        If omitted, `values[:, target_index]` is used.

    target_index:
        Target column index when `target_values` is not provided.

    lookback:
        Number of past timesteps used as input.

    horizon:
        Forecasting horizon. `horizon=1` means one-step-ahead forecasting.

    flatten:
        If True, returns X as shape `(n_samples, lookback * n_features)`.
        If False, returns X as shape `(n_samples, lookback, n_features)`.

    max_samples:
        If provided, keeps the most recent `max_samples` samples. This is useful
        for fast benchmarking on very large datasets.

    return_metadata:
        If True, returns `(X, y, metadata)`.

    Returns
    -------
    X, y or X, y, metadata
    """
    if lookback <= 0:
        raise ValueError("lookback must be strictly positive.")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive.")

    values_2d = _as_2d_values(values)

    if target_values is None:
        if target_index < 0 or target_index >= values_2d.shape[1]:
            raise ValueError(
                f"target_index={target_index} is invalid for "
                f"{values_2d.shape[1]} features."
            )
        target = values_2d[:, target_index]
    else:
        target = _as_1d_target(target_values)

    if len(values_2d) != len(target):
        raise ValueError(
            "values and target_values must have the same length. "
            f"Got {len(values_2d)} and {len(target)}."
        )

    n_timesteps, n_features = values_2d.shape

    last_t = n_timesteps - horizon + 1
    if last_t <= lookback:
        raise ValueError(
            "Not enough timesteps to build windows. "
            f"Need more than lookback+horizon={lookback + horizon}, "
            f"got {n_timesteps}."
        )

    X = []
    y = []
    start_indices = []
    target_indices = []

    for t in range(lookback, last_t):
        X.append(values_2d[t - lookback : t])
        y.append(target[t + horizon - 1])
        start_indices.append(t - lookback)
        target_indices.append(t + horizon - 1)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    start_indices = np.asarray(start_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be strictly positive when provided.")
        if len(y) > max_samples:
            X = X[-max_samples:]
            y = y[-max_samples:]
            start_indices = start_indices[-max_samples:]
            target_indices = target_indices[-max_samples:]

    if flatten:
        X_out = flatten_windows(X)
        last_value_index = lookback * n_features - n_features + target_index
    else:
        X_out = X
        last_value_index = lookback - 1

    metadata = WindowMetadata(
        lookback=lookback,
        horizon=horizon,
        n_features=n_features,
        flattened=flatten,
        last_value_index=last_value_index,
        sample_start_indices=start_indices,
        target_indices=target_indices,
    )

    if return_metadata:
        return X_out, y, metadata

    return X_out, y


def chronological_split(
    X,
    y,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """
    Split data chronologically into train, validation and test partitions.

    This is the recommended split for forecasting. It avoids leakage from the
    future into training data.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}.")

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")

    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1).")

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.")

    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    if n_train <= 0 or n_train + n_val >= n:
        raise ValueError(
            "Invalid split: not enough samples in train/validation/test partitions."
        )

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test


@dataclass(frozen=True)
class StandardizationStats:
    """Train-only standardization parameters."""

    mean: np.ndarray
    std: np.ndarray
    eps: float


def train_only_standardize(
    X_train,
    X_val=None,
    X_test=None,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
):
    """
    Standardize features using training data only.

    For 2D arrays, statistics are computed per column.

    For 3D arrays `(samples, lookback, features)`, statistics are computed per
    feature over both sample and time axes.
    """
    X_train = np.asarray(X_train, dtype=np.float64)

    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train contains NaN or infinite values.")

    if X_train.ndim == 2:
        axis = 0
        keepdims = True
    elif X_train.ndim == 3:
        axis = (0, 1)
        keepdims = True
    else:
        raise ValueError(
            "X_train must be 2D or 3D. "
            f"Received shape {X_train.shape}."
        )

    mean = X_train.mean(axis=axis, keepdims=keepdims)
    std = X_train.std(axis=axis, keepdims=keepdims) + eps

    def transform(X):
        if X is None:
            return None
        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN or infinite values.")
        return (X - mean) / std

    transformed = [transform(X_train)]

    if X_val is not None:
        transformed.append(transform(X_val))

    if X_test is not None:
        transformed.append(transform(X_test))

    if return_stats:
        stats = StandardizationStats(mean=mean, std=std, eps=eps)
        transformed.append(stats)

    if len(transformed) == 1:
        return transformed[0]

    return tuple(transformed)
