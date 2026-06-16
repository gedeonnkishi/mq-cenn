"""
Preprocessing utilities for MQ-CeNN.

This module provides time-series windowing, calendar/phase features, and
seasonal lag helpers. These utilities are intentionally classical and
deterministic: they prepare meaningful temporal representations before the
MQ-CeNN kernel lifting and interference-inspired aggregation stages.
"""

from .windowing import (
    make_supervised_windows,
    chronological_split,
    train_only_standardize,
    flatten_windows,
)

from .calendar_features import (
    make_calendar_features,
    add_calendar_features,
)

from .seasonal_lags import (
    add_seasonal_lag_features,
    make_seasonal_lag_matrix,
    seasonal_naive_forecast,
)

__all__ = [
    "make_supervised_windows",
    "chronological_split",
    "train_only_standardize",
    "flatten_windows",
    "make_calendar_features",
    "add_calendar_features",
    "add_seasonal_lag_features",
    "make_seasonal_lag_matrix",
    "seasonal_naive_forecast",
]
