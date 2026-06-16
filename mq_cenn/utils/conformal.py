from __future__ import annotations

from typing import Tuple

import numpy as np


def quantile_abs_residual(residuals: np.ndarray, coverage: float) -> float:
    """
    Compute the split-conformal absolute residual quantile.

    Parameters
    ----------
    residuals:
        Calibration residuals.
    coverage:
        Target coverage level. Typical values are 0.90 or 0.95.

    Returns
    -------
    float
        Absolute residual radius used for prediction intervals.

    Raises
    ------
    ValueError
        If residuals contain NaN or infinite values.
    """
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)

    if residuals.size == 0:
        return float("nan")

    if not np.isfinite(residuals).all():
        raise ValueError("residuals contains non-finite values.")

    coverage = float(np.clip(coverage, 0.50, 0.999))

    q = np.ceil((residuals.size + 1) * coverage) / residuals.size
    q = min(1.0, q)

    return float(np.quantile(np.abs(residuals), q))


def conformal_interval(
    prediction: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric conformal prediction intervals.

    Parameters
    ----------
    prediction:
        Point predictions.
    radius:
        Conformal absolute residual radius.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper interval bounds.
    """
    prediction = np.asarray(prediction, dtype=np.float64)

    if not np.isfinite(radius):
        lower = np.full_like(prediction, np.nan, dtype=np.float64)
        upper = np.full_like(prediction, np.nan, dtype=np.float64)
        return lower, upper

    radius = float(radius)

    return prediction - radius, prediction + radius


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

_quantile_abs_residual = quantile_abs_residual


__all__ = [
    "quantile_abs_residual",
    "conformal_interval",
    "_quantile_abs_residual",
]
