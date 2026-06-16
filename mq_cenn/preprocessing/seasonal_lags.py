"""
Seasonal lag utilities for time-series forecasting.

Seasonal lags expose values such as t-24, t-48 and t-168 explicitly. This is
essential for hourly electricity, traffic, weather and industrial datasets.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "seasonal_lags requires pandas. Install it with: pip install pandas"
        ) from exc
    return pd


def _validate_lags(lags: Sequence[int]) -> list[int]:
    if not lags:
        raise ValueError("lags must contain at least one positive integer.")

    clean_lags = []
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            raise ValueError("All lags must be strictly positive.")
        clean_lags.append(lag)

    return sorted(set(clean_lags))


def make_seasonal_lag_matrix(
    series,
    *,
    lags: Sequence[int] = (24, 48, 168),
    dropna: bool = True,
    fill_value: float = np.nan,
    return_valid_mask: bool = False,
):
    """
    Create a matrix of seasonal lag features from a univariate series.

    Parameters
    ----------
    series:
        One-dimensional series.

    lags:
        Lag offsets. For hourly data, common values are 24 and 168.

    dropna:
        If True, removes the first rows that do not have all lag values.

    fill_value:
        Value used for missing lag positions when `dropna=False`.

    return_valid_mask:
        If True, returns `(lag_matrix, valid_mask)`.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
    """
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("series contains NaN or infinite values.")

    clean_lags = _validate_lags(lags)
    n = len(y)

    lag_matrix = np.full((n, len(clean_lags)), fill_value, dtype=np.float64)

    for j, lag in enumerate(clean_lags):
        if lag < n:
            lag_matrix[lag:, j] = y[:-lag]

    valid_mask = np.ones(n, dtype=bool)
    if clean_lags:
        valid_mask[: max(clean_lags)] = False

    if dropna:
        lag_matrix = lag_matrix[valid_mask]

    if return_valid_mask:
        return lag_matrix, valid_mask

    return lag_matrix


def add_seasonal_lag_features(
    df,
    *,
    target_col: str,
    lags: Sequence[int] = (24, 48, 168),
    rolling_windows: Sequence[int] = (),
    dropna: bool = True,
    prefix: str = "seasonal",
):
    """
    Append seasonal lag and optional rolling features to a DataFrame.

    Parameters
    ----------
    df:
        pandas DataFrame sorted chronologically.

    target_col:
        Name of the target column.

    lags:
        Seasonal lags to add.

    rolling_windows:
        Optional rolling windows. For each window, both mean and standard
        deviation are added using past information only.

    dropna:
        If True, removes rows with missing lag/rolling values.
    """
    pd = _require_pandas()

    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col!r}.")

    out = df.copy()
    clean_lags = _validate_lags(lags)

    for lag in clean_lags:
        out[f"{prefix}_lag_{lag}"] = out[target_col].shift(lag)

    for window in rolling_windows:
        window = int(window)
        if window <= 1:
            raise ValueError("rolling_windows must contain integers greater than 1.")

        shifted = out[target_col].shift(1)
        out[f"{prefix}_rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{prefix}_rolling_std_{window}"] = shifted.rolling(window).std()

    if dropna:
        out = out.dropna().reset_index(drop=True)

    return out


def seasonal_naive_forecast(
    history,
    *,
    horizon: int = 1,
    seasonal_period: int = 24,
    fallback_to_last: bool = True,
) -> float:
    """
    Return a seasonal-naive forecast from a history window.

    For a history window ending at time t-1, the seasonal naive forecast for
    horizon H uses the value from the same seasonal phase when available.

    Example for hourly data:

    - H=1, period=24 -> approximately same hour yesterday.
    - H=24, period=24 -> last observed value at the same phase.

    If the history is too short and `fallback_to_last=True`, returns the last
    observed value.
    """
    y = np.asarray(history, dtype=np.float64).reshape(-1)

    if len(y) == 0:
        raise ValueError("history must not be empty.")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive.")

    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be strictly positive.")

    # Index relative to the end of the history. For horizon=1 and period=24,
    # this selects y[-24]. For horizon=24 and period=24, this selects y[-1].
    idx = -(seasonal_period - horizon + 1)

    if -len(y) <= idx < len(y):
        return float(y[idx])

    if fallback_to_last:
        return float(y[-1])

    raise ValueError(
        "history is too short for the requested seasonal_period and horizon."
    )
