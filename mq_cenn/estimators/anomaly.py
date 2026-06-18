"""
MQ-CeNN Anomaly Detector
=========================

Unsupervised anomaly detector built directly on top of the MQ-CeNN
reliability calibration mechanism.

The key insight: the ``ReliabilityCalibrator`` was originally built to
decide when to trust the forecast. It does this by combining two signals:

1. **Expert disagreement** — when the pool of kernel experts disagrees
   strongly on a given window, the model is uncertain. Uncertain windows
   are often structurally unusual.

2. **Input novelty** — the ``NoveltyDetector`` computes a robust
   Mahalanobis-style distance from the training distribution. A window
   far from the training regime is a candidate anomaly.

In the forecasting context, low reliability → fallback.
In the detection context, low reliability → anomaly.

The ``score_samples`` method follows the sklearn ``OutlierMixin``
convention: more negative = more anomalous.

Usage
-----
>>> from mq_cenn.estimators.anomaly import MQCeNNAnomalyDetector
>>> from mq_cenn.preprocessing.windowing import make_supervised_windows
>>>
>>> # Fit on normal data only
>>> X_normal, _ = make_supervised_windows(normal_series, lookback=24, horizon=1)
>>> detector = MQCeNNAnomalyDetector()
>>> detector.fit(X_normal)
>>>
>>> # Score new windows — more negative = more anomalous
>>> scores = detector.score_samples(X_test)
>>> labels = detector.predict(X_test)  # +1 normal, -1 anomaly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin

from mq_cenn.core.experts import KernelRidgeExpert, MultiKernelExpertPool
from mq_cenn.core.kernels import DEFAULT_KERNEL_SPECS, KernelSpec
from mq_cenn.core.reliability import ReliabilityCalibrator
from mq_cenn.utils.seed import set_global_seed
from mq_cenn.utils.validation import ArrayLike, _as_2d_float64


@dataclass
class MQCeNNAnomalyTrace:
    """Diagnostics for MQCeNNAnomalyDetector after fitting."""

    n_experts: int
    kernel_families: List[str]
    trained_on_samples: int
    reliability_threshold: float
    mean_reliability_train: float
    anomaly_rate_train: float
    claim_ledger: Dict[str, str] = field(default_factory=dict)


class MQCeNNAnomalyDetector(BaseEstimator, OutlierMixin):
    """
    MQ-CeNN unsupervised anomaly detector.

    Trains on data assumed to be normal (no anomaly labels required).
    Anomaly scores are derived from the reliability signal produced by
    the ``ReliabilityCalibrator``: expert disagreement + input novelty.

    This exploits the ``ReliabilityCalibrator`` that is already part of
    the MQ-CeNN framework, without adding new architectural components.

    No bridge or gate is trained — training is fast and purely classical.

    Parameters
    ----------
    n_features_per_expert : int
        Number of random Fourier features per expert.
    n_experts_per_kernel : int
        Number of experts per kernel family (diversity source).
    kernel_specs : sequence of KernelSpec
        Kernel families to include in the pool.
    alpha_grid : sequence of float
        Ridge regularization values to search via TimeSeriesSplit.
    novelty_weight : float
        Relative weight of the input novelty term in the reliability score.
        Higher values make the detector more sensitive to distribution shift.
        Lower values emphasize expert disagreement only.
    reliability_sensitivity : float
        Controls the steepness of the reliability decay.
        Higher values make the score drop faster for unusual inputs.
    reliability_threshold : float
        Score below which a sample is labeled as anomaly (-1).
        This is used by ``predict()`` only. ``score_samples()`` always
        returns the full continuous score.
    anomaly_rate_train : float
        Expected fraction of anomalies in training data.
        If > 0, the threshold is adjusted upward after fitting so that
        the top ``anomaly_rate_train`` fraction of training samples
        would be flagged. Set to 0.0 to use ``reliability_threshold`` directly.
    random_state : int
        Global seed for reproducibility.

    Notes
    -----
    The detector assumes **training data is predominantly normal**.
    If the training set contains many anomalies, the calibrator will
    treat them as normal and the scores will be poorly calibrated.

    For best results:
    - Use a lookback window consistent with the dominant anomaly duration
      (e.g. if anomalies last 10 timesteps, lookback >= 10).
    - Standardize the input windows using train-only statistics.
    - Fit on a clean reference period before known anomaly events.
    """

    def __init__(
        self,
        n_features_per_expert: int = 256,
        n_experts_per_kernel: int = 2,
        kernel_specs: Sequence[KernelSpec] = DEFAULT_KERNEL_SPECS,
        alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        novelty_weight: float = 0.5,
        reliability_sensitivity: float = 1.0,
        reliability_threshold: float = 0.30,
        anomaly_rate_train: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.n_features_per_expert = int(n_features_per_expert)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.kernel_specs = tuple(kernel_specs)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.novelty_weight = float(novelty_weight)
        self.reliability_sensitivity = float(reliability_sensitivity)
        self.reliability_threshold = float(reliability_threshold)
        self.anomaly_rate_train = float(anomaly_rate_train)
        self.random_state = int(random_state)

        # Fitted attributes
        self.pool_: Optional[MultiKernelExpertPool] = None
        self.reliability_: Optional[ReliabilityCalibrator] = None
        self.threshold_: float = float(reliability_threshold)
        self.trace_: Optional[MQCeNNAnomalyTrace] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.n_features_per_expert < 4:
            raise ValueError("n_features_per_expert must be >= 4.")
        if self.n_experts_per_kernel < 1:
            raise ValueError("n_experts_per_kernel must be >= 1.")
        if not self.kernel_specs:
            raise ValueError("kernel_specs must not be empty.")
        if self.novelty_weight < 0.0:
            raise ValueError("novelty_weight must be non-negative.")
        if self.reliability_sensitivity <= 0.0:
            raise ValueError("reliability_sensitivity must be positive.")
        if not 0.0 < self.reliability_threshold < 1.0:
            raise ValueError("reliability_threshold must be in (0, 1).")
        if not 0.0 <= self.anomaly_rate_train < 0.5:
            raise ValueError("anomaly_rate_train must be in [0, 0.5).")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y=None) -> "MQCeNNAnomalyDetector":
        """
        Fit the anomaly detector on normal data.

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)
            Windows built from a reference period assumed to be normal.
        y : ignored
            Included for sklearn API compatibility.

        Returns
        -------
        self
        """
        self._validate()
        set_global_seed(self.random_state)

        X = _as_2d_float64(X, name="X")
        n = X.shape[0]

        if n < 10:
            raise ValueError(
                "At least 10 samples are required to fit the anomaly detector."
            )

        # ------------------------------------------------------------------
        # Build expert pool.
        #
        # The pool needs targets to fit the ridge heads. Since we are in
        # unsupervised mode, we use a self-supervised proxy: each window
        # predicts its own center value. This is a noise-robust internal
        # reconstruction target.
        #
        # Alternative: use the last value of each window (equivalent to
        # the one-step-ahead proxy). Both work; the center is slightly
        # more robust to edge effects.
        # ------------------------------------------------------------------
        center_col = X.shape[1] // 2
        y_proxy = X[:, center_col]  # self-supervised reconstruction target

        self.pool_ = MultiKernelExpertPool(
            kernel_specs=self.kernel_specs,
            n_experts_per_kernel=self.n_experts_per_kernel,
            n_features_per_expert=self.n_features_per_expert,
            alpha_grid=self.alpha_grid,
            random_state=self.random_state,
        ).fit(X, y_proxy)

        pool_preds = self.pool_.predict_pool(X)  # (n, n_experts)

        # ------------------------------------------------------------------
        # Fit reliability calibrator on the normal training data.
        # ------------------------------------------------------------------
        self.reliability_ = ReliabilityCalibrator(
            disagreement_weight=1.0,
            novelty_weight=self.novelty_weight,
            sensitivity=self.reliability_sensitivity,
        ).fit(X, pool_preds)

        # ------------------------------------------------------------------
        # Calibrate the threshold.
        #
        # If anomaly_rate_train > 0, adjust the threshold so that the
        # top anomaly_rate_train fraction of training samples would be
        # flagged. This is useful when the training set is not perfectly
        # clean and the user knows the approximate contamination rate.
        # ------------------------------------------------------------------
        train_scores = self.reliability_.score(X, pool_preds)

        if self.anomaly_rate_train > 0.0:
            q = 1.0 - float(np.clip(self.anomaly_rate_train, 0.0, 0.49))
            self.threshold_ = float(np.quantile(train_scores, 1.0 - q))
        else:
            self.threshold_ = float(self.reliability_threshold)

        anomaly_rate_train = float((train_scores < self.threshold_).mean())

        self.trace_ = MQCeNNAnomalyTrace(
            n_experts=self.pool_.n_experts_,
            kernel_families=[spec.name for spec in self.kernel_specs],
            trained_on_samples=int(n),
            reliability_threshold=self.threshold_,
            mean_reliability_train=float(train_scores.mean()),
            anomaly_rate_train=anomaly_rate_train,
            claim_ledger={
                "quantum_computation": "No QPU. Classical random-feature proxies only.",
                "anomaly_score": (
                    "Score = -reliability. Reliability = exp(-sensitivity * "
                    "(disagreement + novelty_weight * novelty)). "
                    "More negative = more anomalous."
                ),
                "supervision": (
                    "Unsupervised. No anomaly labels used. "
                    "Self-supervised reconstruction proxy for pool fitting."
                ),
            },
        )

        return self

    # ------------------------------------------------------------------
    # Score and predict
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.pool_ is None or self.reliability_ is None:
            raise RuntimeError("The detector must be fitted before scoring.")

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """
        Compute anomaly scores for input windows.

        Follows the sklearn ``OutlierMixin`` convention:
        **more negative = more anomalous**.

        The raw reliability score in ``[0, 1]`` is negated:
        - A perfectly normal window → reliability ≈ 1 → score ≈ -1 (not anomalous)
        - A strongly anomalous window → reliability ≈ 0 → score ≈ 0 (anomalous)

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Anomaly scores. Range: approximately ``(-1, 0)``.
        """
        self._check_fitted()

        X = _as_2d_float64(X, name="X")

        assert self.pool_ is not None
        assert self.reliability_ is not None

        pool_preds = self.pool_.predict_pool(X)
        reliability = self.reliability_.score(X, pool_preds)

        return -reliability  # sklearn convention: negative = anomalous

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)

        Returns
        -------
        np.ndarray of shape (n_samples,), dtype int
            +1 for normal, -1 for anomaly (sklearn OutlierMixin convention).
        """
        scores = self.score_samples(X)
        # scores = -reliability; threshold_ is a reliability value
        # → sample is anomalous when reliability < threshold_
        # → scores > -threshold_  (since scores = -reliability)
        labels = np.where(scores > -self.threshold_, -1, 1)
        return labels.astype(int)

    def reliability_scores(self, X: ArrayLike) -> np.ndarray:
        """
        Return raw reliability scores in [0, 1].

        Convenience method that inverts the sign convention of
        ``score_samples``. Useful for plotting and thresholding.

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Values in [0, 1]. Low = likely anomaly.
        """
        return -self.score_samples(X)

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """
        Sklearn-compatible decision function.

        Returns the same values as ``score_samples``.
        """
        return self.score_samples(X)


__all__ = [
    "MQCeNNAnomalyDetector",
    "MQCeNNAnomalyTrace",
]
