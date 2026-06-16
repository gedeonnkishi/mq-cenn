from __future__ import annotations

from typing import Optional

import numpy as np

from mq_cenn.utils.validation import _as_2d_float64


class NoveltyDetector:
    """
    Robust diagonal Mahalanobis-like novelty detector.

    The detector uses the median and median absolute deviation style scale,
    making it more robust than a simple mean/std normalization.
    """

    def __init__(self) -> None:
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.ref_: float = 1.0

    def fit(self, X: np.ndarray) -> "NoveltyDetector":
        X = _as_2d_float64(X, name="X")

        self.center_ = np.median(X, axis=0)
        self.scale_ = np.median(np.abs(X - self.center_), axis=0) + 1e-8

        scores = self.score(X)
        self.ref_ = float(np.median(scores) + 1e-8)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("NoveltyDetector must be fitted before score().")

        X = _as_2d_float64(X, name="X")

        if X.shape[1] != self.center_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but detector was fitted "
                f"with {self.center_.shape[0]} features."
            )

        z = (X - self.center_) / self.scale_
        return np.sqrt(np.mean(z * z, axis=1))


class ReliabilityCalibrator:
    """
    Reliability score from expert disagreement and input novelty.

    A score close to 1 means:
    - the input is close to the training regime;
    - experts are relatively consistent.

    A score close to 0 means:
    - the input is novel;
    - experts strongly disagree;
    - fallback may be preferable.
    """

    def __init__(
        self,
        disagreement_weight: float = 1.0,
        novelty_weight: float = 0.5,
        sensitivity: float = 1.0,
    ) -> None:
        self.disagreement_weight = float(disagreement_weight)
        self.novelty_weight = float(novelty_weight)
        self.sensitivity = float(sensitivity)

        if self.disagreement_weight < 0.0:
            raise ValueError("disagreement_weight must be non-negative.")

        if self.novelty_weight < 0.0:
            raise ValueError("novelty_weight must be non-negative.")

        if self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive.")

        self.disagreement_ref_: float = 1.0
        self.novelty_detector_: Optional[NoveltyDetector] = None

    def fit(
        self,
        X_ref: np.ndarray,
        pool_preds_ref: np.ndarray,
    ) -> "ReliabilityCalibrator":
        X_ref = _as_2d_float64(X_ref, name="X_ref")
        pool_preds_ref = _as_2d_float64(pool_preds_ref, name="pool_preds_ref")

        if X_ref.shape[0] != pool_preds_ref.shape[0]:
            raise ValueError("X_ref and pool_preds_ref length mismatch.")

        disagreement = np.var(pool_preds_ref, axis=1)
        self.disagreement_ref_ = float(np.median(disagreement) + 1e-8)

        self.novelty_detector_ = NoveltyDetector().fit(X_ref)

        return self

    def score(self, X: np.ndarray, pool_preds: np.ndarray) -> np.ndarray:
        if self.novelty_detector_ is None:
            raise RuntimeError("ReliabilityCalibrator must be fitted before score().")

        X = _as_2d_float64(X, name="X")
        pool_preds = _as_2d_float64(pool_preds, name="pool_preds")

        if X.shape[0] != pool_preds.shape[0]:
            raise ValueError("X and pool_preds length mismatch.")

        disagreement = np.var(pool_preds, axis=1) / self.disagreement_ref_
        novelty = self.novelty_detector_.score(X) / self.novelty_detector_.ref_

        energy = (
            self.disagreement_weight * disagreement
            + self.novelty_weight * novelty
        )

        reliability = np.exp(-self.sensitivity * energy)
        return np.clip(reliability, 0.0, 1.0)


__all__ = [
    "NoveltyDetector",
    "ReliabilityCalibrator",
]
