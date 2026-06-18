"""
MQ-CeNN Multi-Step Regressor
=============================

Direct multi-output extension of MQCeNNRegressor for long-horizon forecasting.

Architecture
------------
Each KernelRidgeExpert predicts the full horizon vector simultaneously
(multi-output ridge regression). The CrossExpertBridge and SignedInterferenceGate
operate on per-expert predictions, then combine them to produce the final
(n_samples, horizon) forecast matrix.

The gate assigns one weight per expert (shared across all horizon steps), which
forces the model to select experts that are globally coherent over the full
prediction horizon — a deliberate regularization choice.

Fallback and reliability operate per sample, collapsing the horizon axis
using the mean disagreement across steps.

Usage
-----
>>> from mq_cenn.estimators.multistep import MQCeNNMultiStepRegressor
>>> from mq_cenn.preprocessing.windowing import make_multistep_windows, chronological_split, train_only_standardize
>>>
>>> X, y = make_multistep_windows(series, lookback=96, horizon=24)
>>> X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)
>>> X_train, X_val, X_test = train_only_standardize(X_train, X_val, X_test)
>>>
>>> model = MQCeNNMultiStepRegressor(horizon=24)
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)  # shape (n_test, 24)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

from mq_cenn.backends.dispatcher import BackendInfo, resolve_backend
from mq_cenn.core.bridge import CrossExpertBridge
from mq_cenn.core.gate import SignedInterferenceGate
from mq_cenn.core.kernels import DEFAULT_KERNEL_SPECS, KernelSpec
from mq_cenn.core.reliability import ReliabilityCalibrator
from mq_cenn.utils.seed import set_global_seed
from mq_cenn.utils.validation import ArrayLike, _as_2d_float64, _safe_std


FallbackStrategy = Literal["teacher_mean", "persistence", "stable_ridge"]


# ---------------------------------------------------------------------------
# Multi-output kernel expert
# ---------------------------------------------------------------------------


class _MultiOutputKernelRidgeExpert:
    """
    One random-feature multi-output ridge expert.

    The spectral projector is the same as in the single-step case.
    The ridge head solves for a (n_features_rff, horizon) weight matrix
    using a single Cholesky / solve per expert (not one solve per step).
    """

    def __init__(
        self,
        spec: KernelSpec,
        n_features: int,
        alpha: float,
        random_state: int,
    ) -> None:
        from mq_cenn.core.kernels import SpectralFeatureProjector

        self.spec = spec
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

        self.projector_ = SpectralFeatureProjector(
            spec=spec,
            n_features=self.n_features,
            random_state=self.random_state,
        )
        self.beta_: Optional[np.ndarray] = None  # (n_rff, horizon)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "_MultiOutputKernelRidgeExpert":
        """
        Fit multi-output ridge.

        Parameters
        ----------
        X : (n_samples, n_input_features)
        Y : (n_samples, horizon)
        """
        X = _as_2d_float64(X, name="X")
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (n_samples, horizon), got shape {Y.shape}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y row count mismatch.")

        self.projector_.fit(X.shape[1])
        Z = self.projector_.transform(X)  # (n, n_rff)

        A = Z.T @ Z
        A.flat[:: A.shape[0] + 1] += self.alpha  # ridge penalty

        try:
            self.beta_ = np.linalg.solve(A, Z.T @ Y)  # (n_rff, horizon)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(A, Z.T @ Y, rcond=None)[0]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, horizon) predictions."""
        if self.beta_ is None:
            raise RuntimeError("Expert must be fitted before predict().")
        X = _as_2d_float64(X, name="X")
        return self.projector_.transform(X) @ self.beta_  # (n, horizon)


class _MultiOutputExpertPool:
    """
    Pool of multi-output kernel experts.

    ``predict_pool`` returns shape ``(n_samples, n_experts, horizon)``,
    i.e. one (n_experts, horizon) block per sample.
    """

    def __init__(
        self,
        kernel_specs: Sequence[KernelSpec],
        n_experts_per_kernel: int,
        n_features_per_expert: int,
        alpha: float,
        random_state: int,
    ) -> None:
        self.kernel_specs = tuple(kernel_specs)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.n_features_per_expert = int(n_features_per_expert)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.experts_: List[_MultiOutputKernelRidgeExpert] = []

    @property
    def n_experts_(self) -> int:
        return len(self.experts_)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "_MultiOutputExpertPool":
        self.experts_ = []
        seed = 0
        for spec in self.kernel_specs:
            for _ in range(self.n_experts_per_kernel):
                expert = _MultiOutputKernelRidgeExpert(
                    spec=spec,
                    n_features=self.n_features_per_expert,
                    alpha=self.alpha,
                    random_state=self.random_state + 9973 * seed,
                )
                expert.fit(X, Y)
                self.experts_.append(expert)
                seed += 1
        return self

    def predict_pool(self, X: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray of shape (n_samples, n_experts, horizon)
        """
        if not self.experts_:
            raise RuntimeError("Pool must be fitted before predict_pool().")
        preds = [e.predict(X) for e in self.experts_]  # list of (n, horizon)
        return np.stack(preds, axis=1)  # (n, n_experts, horizon)


# ---------------------------------------------------------------------------
# Dataclass for diagnostics
# ---------------------------------------------------------------------------


@dataclass
class MQCeNNMultiStepTrace:
    """Training diagnostics for MQCeNNMultiStepRegressor."""

    horizon: int
    best_val_loss: float
    epochs_ran: int
    n_experts: int
    kernel_families: List[str]
    alpha_used: float
    reliability_threshold: float
    mean_reliability_cal: float
    fallback_rate_cal: float
    trained_on_samples: int
    calibrated_on_samples: int
    backend: str = "numpy"
    device: str = "cpu"
    claim_ledger: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class MQCeNNMultiStepRegressor(BaseEstimator, RegressorMixin):
    """
    MQ-CeNN direct multi-step forecaster.

    Predicts ``horizon`` future steps simultaneously from a lookback window.
    This avoids the error accumulation of recursive one-step-ahead strategies.

    The architecture is identical to ``MQCeNNRegressor`` except:

    - Each KernelRidgeExpert is replaced by a multi-output ridge head that
      directly learns the full (horizon,) target vector.
    - The pool produces shape ``(n_samples, n_experts, horizon)``; the gate
      assigns per-expert scalar weights, broadcast across the horizon axis.
    - Reliability is computed on the mean disagreement across horizon steps,
      collapsing the horizon dimension before the ReliabilityCalibrator.

    Parameters
    ----------
    horizon : int
        Number of future timesteps to forecast simultaneously.
    n_features_per_expert : int
        Number of random Fourier features per expert.
    n_experts_per_kernel : int
        Number of experts per kernel family.
    kernel_specs : sequence of KernelSpec
        Kernel families to include in the pool.
    alpha : float
        Ridge regularization parameter (no cross-validation — use a fixed
        value chosen from prior knowledge or grid-search externally).
    bridge_dim : int
        Dimension of the CrossExpertBridge representation.
    cenn_hidden : int
        Hidden size of the SignedInterferenceGate temporal encoder.
    cenn_kernel : int
        Kernel size of the 1D temporal convolution inside the gate.
    cenn_epochs : int
        Maximum training epochs for bridge + gate.
    cenn_lr : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Mini-batch size for gate training.
    patience : int
        Early-stopping patience (epochs without improvement on val loss).
    calibration_fraction : float
        Fraction of training data held out for calibration (chronological).
    signed_interference : bool
        If True, use signed L1-normalized weights (interference-like).
        If False, use softmax weights (ablation baseline).
    reliability_threshold : float
        Samples with reliability below this value trigger fallback.
    reliability_sensitivity : float
        Exponential decay sensitivity in the reliability score.
    novelty_weight : float
        Weight of the input novelty term in the reliability score.
    fallback_strategy : str
        ``"teacher_mean"`` | ``"persistence"``.
        ``"stable_ridge"`` is not supported in multi-step mode (use teacher_mean).
    random_state : int
        Global seed for reproducibility.
    backend : str
        ``"auto"`` | ``"numpy"`` | ``"torch"``.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"``.
    """

    def __init__(
        self,
        horizon: int = 24,
        n_features_per_expert: int = 512,
        n_experts_per_kernel: int = 2,
        kernel_specs: Sequence[KernelSpec] = DEFAULT_KERNEL_SPECS,
        alpha: float = 1.0,
        bridge_dim: int = 32,
        cenn_hidden: int = 64,
        cenn_kernel: int = 3,
        cenn_epochs: int = 40,
        cenn_lr: float = 1e-3,
        batch_size: int = 256,
        patience: int = 6,
        calibration_fraction: float = 0.15,
        signed_interference: bool = True,
        reliability_threshold: float = 0.30,
        reliability_sensitivity: float = 1.0,
        novelty_weight: float = 0.5,
        fallback_strategy: FallbackStrategy = "teacher_mean",
        random_state: int = 42,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.horizon = int(horizon)
        self.n_features_per_expert = int(n_features_per_expert)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.kernel_specs = tuple(kernel_specs)
        self.alpha = float(alpha)
        self.bridge_dim = int(bridge_dim)
        self.cenn_hidden = int(cenn_hidden)
        self.cenn_kernel = int(cenn_kernel)
        self.cenn_epochs = int(cenn_epochs)
        self.cenn_lr = float(cenn_lr)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.calibration_fraction = float(calibration_fraction)
        self.signed_interference = bool(signed_interference)
        self.reliability_threshold = float(reliability_threshold)
        self.reliability_sensitivity = float(reliability_sensitivity)
        self.novelty_weight = float(novelty_weight)
        self.fallback_strategy = fallback_strategy
        self.random_state = int(random_state)
        self.backend = backend
        self.device = device

        # Fitted attributes
        self.backend_info_: Optional[BackendInfo] = None
        self.pool_: Optional[_MultiOutputExpertPool] = None
        self.bridge_: Optional[CrossExpertBridge] = None
        self.gate_: Optional[SignedInterferenceGate] = None
        self.reliability_: Optional[ReliabilityCalibrator] = None
        self.trace_: Optional[MQCeNNMultiStepTrace] = None

        self.pool_mean_: Optional[np.ndarray] = None  # (1, n_experts)
        self.pool_std_: Optional[np.ndarray] = None   # (1, n_experts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        if self.backend_info_ is None:
            self.backend_info_ = resolve_backend(self.backend, self.device)
        return torch.device(self.backend_info_.device)

    def _validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.n_features_per_expert < 4:
            raise ValueError("n_features_per_expert must be >= 4.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if not 0.0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5).")
        if self.fallback_strategy == "stable_ridge":
            raise ValueError(
                "fallback_strategy='stable_ridge' is not supported in multi-step mode. "
                "Use 'teacher_mean' or 'persistence'."
            )

    def _chronological_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 20:
            raise ValueError("At least 20 samples are required.")
        frac = float(np.clip(self.calibration_fraction, 0.05, 0.40))
        cut = int(np.floor(n * (1.0 - frac)))
        cut = min(max(cut, 5), n - 5)
        idx = np.arange(n)
        return idx[:cut], idx[cut:]

    def _pool_to_scalar(self, pool_preds: np.ndarray) -> np.ndarray:
        """
        Collapse (n_samples, n_experts, horizon) → (n_samples, n_experts)
        by averaging over the horizon axis.

        Used to feed a scalar-per-expert signal to ReliabilityCalibrator
        and to compute pool normalization statistics, both of which were
        designed for 1-step (scalar) experts.
        """
        return pool_preds.mean(axis=2)  # (n, n_experts)

    def _normalize_pool(self, pool_scalar: np.ndarray) -> np.ndarray:
        """Normalize collapsed expert predictions."""
        if self.pool_mean_ is None or self.pool_std_ is None:
            raise RuntimeError("Pool normalization not fitted.")
        return (pool_scalar - self.pool_mean_) / self.pool_std_

    def _context(self, X: np.ndarray) -> np.ndarray:
        """Build (n, seq_len, 1) context tensor for the gate."""
        X = _as_2d_float64(X, name="X")
        n, w = X.shape
        return X.reshape(n, w, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, Y: ArrayLike) -> "MQCeNNMultiStepRegressor":
        """
        Fit the multi-step regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)
            Input windows produced by ``make_multistep_windows``.
        Y : array-like of shape (n_samples, horizon)
            Multi-step target matrix.

        Returns
        -------
        self
        """
        self._validate()
        set_global_seed(self.random_state)
        self.backend_info_ = resolve_backend(self.backend, self.device)

        X = _as_2d_float64(X, name="X")
        Y = np.asarray(Y, dtype=np.float64)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)  # tolerate single-step targets

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (n_samples, horizon), got {Y.shape}.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y row count mismatch.")

        if Y.shape[1] != self.horizon:
            raise ValueError(
                f"Y has {Y.shape[1]} columns but horizon={self.horizon}. "
                "Either adjust horizon or re-build windows."
            )

        train_idx, cal_idx = self._chronological_split(X.shape[0])
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_cal, Y_cal = X[cal_idx], Y[cal_idx]

        # --- Expert pool ---
        self.pool_ = _MultiOutputExpertPool(
            kernel_specs=self.kernel_specs,
            n_experts_per_kernel=self.n_experts_per_kernel,
            n_features_per_expert=self.n_features_per_expert,
            alpha=self.alpha,
            random_state=self.random_state,
        ).fit(X_train, Y_train)

        # pool_preds: (n, n_experts, horizon)
        pool_train = self.pool_.predict_pool(X_train)
        pool_cal = self.pool_.predict_pool(X_cal)

        # Scalar summary per expert for bridge / reliability
        pool_train_s = self._pool_to_scalar(pool_train)  # (n, n_experts)
        pool_cal_s = self._pool_to_scalar(pool_cal)

        self.pool_mean_ = pool_train_s.mean(axis=0, keepdims=True)
        self.pool_std_ = _safe_std(pool_train_s, axis=0, keepdims=True)

        # --- Reliability calibrator (uses scalar summary) ---
        self.reliability_ = ReliabilityCalibrator(
            disagreement_weight=1.0,
            novelty_weight=self.novelty_weight,
            sensitivity=self.reliability_sensitivity,
        ).fit(X_train, pool_train_s)

        # --- Gate training ---
        self._fit_gate(
            X_train=X_train,
            Y_train=Y_train,
            pool_train=pool_train,
            pool_train_s=pool_train_s,
            X_cal=X_cal,
            Y_cal=Y_cal,
            pool_cal=pool_cal,
            pool_cal_s=pool_cal_s,
        )

        # --- Diagnostics ---
        cal_reliability = self.reliability_.score(X_cal, pool_cal_s)
        fallback_rate_cal = float((cal_reliability < self.reliability_threshold).mean())

        assert self.backend_info_ is not None
        assert self.pool_ is not None

        self.trace_ = MQCeNNMultiStepTrace(
            horizon=self.horizon,
            best_val_loss=float(self._best_val_loss_),
            epochs_ran=int(self._epochs_ran_),
            n_experts=self.pool_.n_experts_,
            kernel_families=[spec.name for spec in self.kernel_specs],
            alpha_used=self.alpha,
            reliability_threshold=float(self.reliability_threshold),
            mean_reliability_cal=float(cal_reliability.mean()),
            fallback_rate_cal=fallback_rate_cal,
            trained_on_samples=int(len(train_idx)),
            calibrated_on_samples=int(len(cal_idx)),
            backend=self.backend_info_.backend,
            device=self.backend_info_.device,
            claim_ledger={
                "quantum_computation": "No QPU. Classical random-feature proxies only.",
                "multi_step": (
                    "Direct multi-output prediction. "
                    "Gate weights are shared across horizon steps."
                ),
            },
        )

        return self

    def _fit_gate(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        pool_train: np.ndarray,
        pool_train_s: np.ndarray,
        X_cal: np.ndarray,
        Y_cal: np.ndarray,
        pool_cal: np.ndarray,
        pool_cal_s: np.ndarray,
    ) -> None:
        """Train CrossExpertBridge + SignedInterferenceGate for multi-step output."""
        dev = self._device()
        n_experts = pool_train.shape[1]

        X_ctx_train = self._context(X_train)
        X_ctx_cal = self._context(X_cal)

        pn_train = self._normalize_pool(pool_train_s).astype(np.float32)
        pn_cal = self._normalize_pool(pool_cal_s).astype(np.float32)

        # Tensors
        x_train_t = torch.as_tensor(X_ctx_train, device=dev)
        pn_train_t = torch.as_tensor(pn_train, device=dev)
        # pool_train_t: (n, n_experts, horizon) — for prediction
        pool_train_t = torch.as_tensor(pool_train.astype(np.float32), device=dev)
        Y_train_t = torch.as_tensor(Y_train.astype(np.float32), device=dev)

        x_cal_t = torch.as_tensor(X_ctx_cal, device=dev)
        pn_cal_t = torch.as_tensor(pn_cal, device=dev)
        pool_cal_t = torch.as_tensor(pool_cal.astype(np.float32), device=dev)
        Y_cal_t = torch.as_tensor(Y_cal.astype(np.float32), device=dev)

        self.bridge_ = CrossExpertBridge(
            n_experts=n_experts,
            bridge_dim=self.bridge_dim,
            dropout=0.05,
        ).to(dev)

        self.gate_ = SignedInterferenceGate(
            context_channels=X_ctx_train.shape[2],
            n_experts=n_experts,
            bridge_dim=self.bridge_dim,
            hidden_dim=self.cenn_hidden,
            kernel_size=self.cenn_kernel,
            dropout=0.05,
            signed=self.signed_interference,
        ).to(dev)

        params = list(self.bridge_.parameters()) + list(self.gate_.parameters())
        optimizer = optim.AdamW(params, lr=self.cenn_lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_bridge_sd = None
        best_gate_sd = None
        best_val = np.inf
        no_gain = 0
        epoch = -1

        order_base = np.arange(X_train.shape[0])

        for epoch in range(max(0, self.cenn_epochs)):
            self.bridge_.train()
            self.gate_.train()

            order = order_base.copy()
            np.random.default_rng(self.random_state + epoch).shuffle(order)

            for start in range(0, len(order), self.batch_size):
                ids_np = order[start : start + self.batch_size]
                ids = torch.as_tensor(ids_np, dtype=torch.long, device=dev)

                optimizer.zero_grad()

                bridge = self.bridge_(pn_train_t[ids])          # (b, bridge_dim)
                weights = self.gate_(x_train_t[ids], bridge)    # (b, n_experts)

                # Weighted sum: (b, n_experts) x (b, n_experts, horizon) → (b, horizon)
                pred = (weights.unsqueeze(2) * pool_train_t[ids]).sum(dim=1)

                loss = loss_fn(pred, Y_train_t[ids])
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            # Validation
            self.bridge_.eval()
            self.gate_.eval()

            with torch.no_grad():
                bridge_cal = self.bridge_(pn_cal_t)
                weights_cal = self.gate_(x_cal_t, bridge_cal)
                pred_cal = (weights_cal.unsqueeze(2) * pool_cal_t).sum(dim=1)
                val_loss = float(loss_fn(pred_cal, Y_cal_t).detach().cpu())

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_bridge_sd = {k: v.detach().cpu().clone() for k, v in self.bridge_.state_dict().items()}
                best_gate_sd = {k: v.detach().cpu().clone() for k, v in self.gate_.state_dict().items()}
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= self.patience:
                    break

        if best_bridge_sd is not None:
            self.bridge_.load_state_dict(best_bridge_sd)
        if best_gate_sd is not None:
            self.gate_.load_state_dict(best_gate_sd)

        self._best_val_loss_ = float(best_val) if np.isfinite(best_val) else float("nan")
        self._epochs_ran_ = int(max(0, epoch + 1))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if (
            self.pool_ is None
            or self.bridge_ is None
            or self.gate_ is None
            or self.reliability_ is None
        ):
            raise RuntimeError("The model must be fitted before prediction.")

    def _predict_core(
        self, X: np.ndarray, pool_preds: np.ndarray, pool_scalar: np.ndarray
    ) -> np.ndarray:
        """
        Core gated prediction.

        Parameters
        ----------
        X : (n, window_size)
        pool_preds : (n, n_experts, horizon)
        pool_scalar : (n, n_experts)  — mean across horizon, for normalization

        Returns
        -------
        np.ndarray of shape (n, horizon)
        """
        dev = self._device()
        X_ctx = self._context(X)
        pn = self._normalize_pool(pool_scalar).astype(np.float32)

        assert self.bridge_ is not None
        assert self.gate_ is not None

        self.bridge_.eval()
        self.gate_.eval()

        with torch.no_grad():
            x_t = torch.as_tensor(X_ctx, device=dev)
            pn_t = torch.as_tensor(pn, device=dev)
            pool_t = torch.as_tensor(pool_preds.astype(np.float32), device=dev)

            bridge = self.bridge_(pn_t)
            weights = self.gate_(x_t, bridge)                     # (n, n_experts)
            pred = (weights.unsqueeze(2) * pool_t).sum(dim=1)    # (n, horizon)

        return pred.detach().cpu().numpy().astype(np.float64)

    def _fallback_prediction(
        self,
        X: np.ndarray,
        pool_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Fallback for low-reliability samples.

        Returns (n_samples, horizon).
        """
        if self.fallback_strategy == "persistence":
            # Repeat last value in window across all horizon steps
            last_val = X[:, -1].reshape(-1, 1)  # (n, 1)
            return np.repeat(last_val, self.horizon, axis=1)

        # teacher_mean: average of expert forecasts
        return pool_preds.mean(axis=1)  # (n, horizon)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict ``horizon`` steps ahead for each input window.

        Parameters
        ----------
        X : array-like of shape (n_samples, window_size)

        Returns
        -------
        np.ndarray of shape (n_samples, horizon)
        """
        self._check_fitted()

        X = _as_2d_float64(X, name="X")

        assert self.pool_ is not None
        assert self.reliability_ is not None

        pool_preds = self.pool_.predict_pool(X)       # (n, n_experts, horizon)
        pool_scalar = self._pool_to_scalar(pool_preds) # (n, n_experts)

        reliability = self.reliability_.score(X, pool_scalar)
        pred = self._predict_core(X, pool_preds, pool_scalar)

        # Apply fallback to low-reliability samples
        mask = reliability < self.reliability_threshold
        if np.any(mask):
            pred[mask] = self._fallback_prediction(X[mask], pool_preds[mask])

        return pred

    def predict_with_diagnostics(self, X: ArrayLike) -> dict:
        """
        Return prediction plus per-sample diagnostics.

        Returns
        -------
        dict with keys:
            ``prediction``   (n, horizon)
            ``reliability``  (n,) — scalar reliability score per sample
            ``fallback_mask`` (n,) — True where fallback was used
            ``teacher_mean`` (n, horizon) — unweighted expert average
            ``pool_predictions`` (n, n_experts, horizon)
        """
        self._check_fitted()

        X = _as_2d_float64(X, name="X")

        assert self.pool_ is not None
        assert self.reliability_ is not None

        pool_preds = self.pool_.predict_pool(X)
        pool_scalar = self._pool_to_scalar(pool_preds)
        reliability = self.reliability_.score(X, pool_scalar)

        pred = self._predict_core(X, pool_preds, pool_scalar)
        fallback_mask = reliability < self.reliability_threshold

        if np.any(fallback_mask):
            pred[fallback_mask] = self._fallback_prediction(
                X[fallback_mask], pool_preds[fallback_mask]
            )

        return {
            "prediction": pred,
            "reliability": reliability,
            "fallback_mask": fallback_mask.astype(bool),
            "teacher_mean": pool_preds.mean(axis=1),
            "pool_predictions": pool_preds,
        }


__all__ = [
    "MQCeNNMultiStepRegressor",
    "MQCeNNMultiStepTrace",
]
