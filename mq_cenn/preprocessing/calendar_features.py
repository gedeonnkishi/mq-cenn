"""
Calendar and cyclic time features for time-series forecasting.

These features are important for datasets with hourly, daily, weekly or annual
seasonality. They provide explicit phase information before kernel lifting.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "calendar_features requires pandas. Install it with: pip install pandas"
        ) from exc
    return pd


def _sin_cos(values: np.ndarray, period: float):
    angle = 2.0 * np.pi * values.astype(np.float64) / float(period)
    return np.sin(angle), np.cos(angle)


def make_calendar_features(
    timestamps,
    *,
    include_hour: bool = True,
    include_dayofweek: bool = True,
    include_dayofmonth: bool = False,
    include_dayofyear: bool = True,
    include_month: bool = True,
    include_weekend: bool = True,
    include_quarter: bool = False,
    cyclic: bool = True,
    return_names: bool = False,
):
    """
    Build calendar features from timestamps.

    Parameters
    ----------
    timestamps:
        Array-like timestamps accepted by `pandas.to_datetime`.

    cyclic:
        If True, encodes periodic fields with sin/cos pairs.
        If False, uses normalized scalar values.

    return_names:
        If True, returns `(features, names)`.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, list[str]]
    """
    pd = _require_pandas()

    ts = pd.to_datetime(timestamps, errors="coerce")
    if ts.isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    features = []
    names = []

    def add_cyclic_or_scalar(name: str, values, period: float):
        values_arr = np.asarray(values, dtype=np.float64)
        if cyclic:
            s, c = _sin_cos(values_arr, period)
            features.append(s)
            features.append(c)
            names.append(f"{name}_sin")
            names.append(f"{name}_cos")
        else:
            features.append(values_arr / float(period))
            names.append(name)

    if include_hour:
        add_cyclic_or_scalar("hour", ts.hour, 24.0)

    if include_dayofweek:
        add_cyclic_or_scalar("dayofweek", ts.dayofweek, 7.0)

    if include_dayofmonth:
        # Day of month starts at 1. Use 31 as conservative period.
        add_cyclic_or_scalar("dayofmonth", ts.day - 1, 31.0)

    if include_dayofyear:
        # Use 366 to remain safe for leap years.
        add_cyclic_or_scalar("dayofyear", ts.dayofyear - 1, 366.0)

    if include_month:
        add_cyclic_or_scalar("month", ts.month - 1, 12.0)

    if include_weekend:
        weekend = (ts.dayofweek >= 5).astype(np.float64)
        features.append(np.asarray(weekend, dtype=np.float64))
        names.append("is_weekend")

    if include_quarter:
        quarter = (ts.quarter - 1).astype(np.float64)
        add_cyclic_or_scalar("quarter", quarter, 4.0)

    if not features:
        X = np.empty((len(ts), 0), dtype=np.float64)
    else:
        X = np.column_stack(features).astype(np.float64)

    if return_names:
        return X, names

    return X


def add_calendar_features(
    df,
    *,
    time_col: str,
    drop_time: bool = False,
    prefix: str = "cal",
    cyclic: bool = True,
):
    """
    Return a copy of a DataFrame with calendar features appended.

    Parameters
    ----------
    df:
        pandas DataFrame.

    time_col:
        Timestamp column name.

    drop_time:
        If True, removes the original timestamp column.

    prefix:
        Prefix applied to generated feature names.
    """
    pd = _require_pandas()

    if time_col not in df.columns:
        raise ValueError(f"Missing time column {time_col!r}.")

    out = df.copy()

    X_cal, names = make_calendar_features(
        out[time_col],
        cyclic=cyclic,
        return_names=True,
    )

    for i, name in enumerate(names):
        out[f"{prefix}_{name}"] = X_cal[:, i]

    if drop_time:
        out = out.drop(columns=[time_col])

    return out
