from .windowing import (
    WindowMetadata,
    StandardizationStats,
    make_supervised_windows,
    make_multistep_windows,
    chronological_split,
    train_only_standardize,
    flatten_windows,
)

try:
    from .calendar_features import make_calendar_features, add_calendar_features
except Exception:  # pragma: no cover - optional pandas-dependent imports
    make_calendar_features = None
    add_calendar_features = None

try:
    from .seasonal_lags import (
        add_seasonal_lag_features,
        make_seasonal_lag_matrix,
        seasonal_naive_forecast,
    )
except Exception:  # pragma: no cover - optional pandas-dependent imports
    add_seasonal_lag_features = None
    make_seasonal_lag_matrix = None
    seasonal_naive_forecast = None


__all__ = [
    "WindowMetadata",
    "StandardizationStats",
    "make_supervised_windows",
    "make_multistep_windows",
    "chronological_split",
    "train_only_standardize",
    "flatten_windows",
    "make_calendar_features",
    "add_calendar_features",
    "add_seasonal_lag_features",
    "make_seasonal_lag_matrix",
    "seasonal_naive_forecast",
]
