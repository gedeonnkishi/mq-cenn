from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

from mq_cenn.backends.dispatcher import BackendInfo, resolve_backend
from mq_cenn.core.bridge import CrossExpertBridge
from mq_cenn.core.experts import KernelRidgeExpert, MultiKernelExpertPool
from mq_cenn.core.gate import SignedInterferenceGate
from mq_cenn.core.kernels import DEFAULT_KERNEL_SPECS, KernelSpec
from mq_cenn.core.reliability import ReliabilityCalibrator
from mq_cenn.utils.conformal import _quantile_abs_residual
from mq_cenn.utils.seed import set_global_seed
from mq_cenn.utils.validation import ArrayLike, _as_1d_float64, _as_2d_float64, _safe_std


FallbackStrategy = Literal["stable_ridge", "teacher_mean", "persistence"]


@dataclass
class MQCeNNTrace:
    """
    Training and calibration diagnostics for MQ-CeNN.
    """

    best_val_loss: float
    epochs_ran: int
    n_experts: int
    kernel_families: List[str]
    best_alpha_per_kernel: Dict[str, float]
    reliability_threshold: float
    mean_reliability_cal: float
    fallback_rate_cal: float
    conformal_coverage: float
    conformal_abs_radius: float
    trained_on_samples: int
    calibrated_on_samples: int
    backend: str = "numpy"
    device: str = "cpu"
    claim_ledger: Dict[str, str] = field(default_factory=dict)


class MQCeNNRegressor(BaseEstimator, RegressorMixin):
    """
    MQ-CeNN regressor for time-series forecasting.

    The model combines:
    - heterogeneous kernel-inspired random-feature experts;
    - cross-expert interaction bridge;
    - signed CeNN-inspired temporal gate;
    - reliability calibration;
    - fallback prediction;
    - conformal prediction intervals.

    This implementation remains classical and claim-bounded.
    It does not implement physical quantum computation.
    """

    def __init__(
        self,
        n_features_per_expert: int = 512,
        n_experts_per_kernel: int = 2,
        kernel_specs: Sequence[KernelSpec] = DEFAULT_KERNEL_SPECS,
        alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        bridge_dim: int = 32,
        cenn_hidden: int = 64,
        cenn_kernel: int = 3,
        cenn_epochs: int = 40,
        cenn_lr: float = 1e-3,
        batch_size: int = 512,
        patience: int = 6,
        calibration_fraction: float = 0.15,
        signed_interference: bool = True,
        reliability_threshold: float = 0.30,
        reliability_sensitivity: float = 1.0,
        novelty_weight: float = 0.5,
        fallback_strategy: FallbackStrategy = "stable_ridge",
        conformal_coverage: float = 0.90,
        stationarize: bool = False,
        last_value_index: Optional[int] = None,
        random_state: int = 42,
        backend: str = "auto",
        device: str = "auto",
        # Backward-compatible aliases used by earlier notebooks.
        n_quantum_features: Optional[int] = None,
        n_experts: Optional[int] = None,
        base_seed: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> None:
        if n_quantum_features is not None:
            n_features_per_expert = int(n_quantum_features)

        if base_seed is not None:
            random_state = int(base_seed)

        if gamma is not None:
            g = float(gamma)
            kernel_specs = tuple(
                KernelSpec(spec.name, spec.gamma * g, spec.period, spec.degree)
                for spec in kernel_specs
            )

        if n_experts is not None:
            n_families = max(1, len(tuple(kernel_specs)))
            n_experts_per_kernel = max(
                1,
                int(np.ceil(int(n_experts) / n_families)),
            )

        self.n_features_per_expert = int(n_features_per_expert)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.kernel_specs = tuple(kernel_specs)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
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
        self.conformal_coverage = float(conformal_coverage)
        self.stationarize = bool(stationarize)
        self.last_value_index = last_value_index
        self.random_state = int(random_state)
        self.backend = backend
        self.device = device

        # Legacy aliases kept so sklearn.get_params() remains valid.
        self.n_quantum_features = n_quantum_features
        self.n_experts = n_experts
        self.base_seed = base_seed
        self.gamma = gamma

        self.backend_info_: Optional[BackendInfo] = None
        self.pool_: Optional[MultiKernelExpertPool] = None
        self.fallback_: Optional[KernelRidgeExpert] = None
        self.bridge_: Optional[CrossExpertBridge] = None
        self.gate_: Optional[SignedInterferenceGate] = None
        self.reliability_: Optional[ReliabilityCalibrator] = None
        self.trace_: Optional[MQCeNNTrace] = None

        self.pool_mean_: Optional[np.ndarray] = None
        self.pool_std_: Optional[np.ndarray] = None
        self.conformal_abs_radius_: float = float("nan")

    def _resolve_backend(self) -> BackendInfo:
        self.backend_info_ = resolve_backend(self.backend, self.device)
        return self.backend_info_

    def _device(self) -> torch.device:
        if self.backend_info_ is None:
            self._resolve_backend()

        assert self.backend_info_ is not None
        return torch.device(self.backend_info_.device)

    def _validate_hyperparameters(self) -> None:
        if self.n_features_per_expert < 4:
            raise ValueError("n_features_per_expert must be >= 4.")

        if self.n_experts_per_kernel < 1:
            raise ValueError("n_experts_per_kernel must be >= 1.")

        if self.bridge_dim < 1:
            raise ValueError("bridge_dim must be >= 1.")

        if self.cenn_hidden < 1:
            raise ValueError("cenn_hidden must be >= 1.")

        if self.cenn_kernel < 1:
            raise ValueError("cenn_kernel must be >= 1.")

        if self.cenn_epochs < 0:
            raise ValueError("cenn_epochs must be >= 0.")

        if self.cenn_lr <= 0:
            raise ValueError("cenn_lr must be positive.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if self.patience < 1:
            raise ValueError("patience must be >= 1.")

        if not 0.0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5).")

        if self.fallback_strategy not in {
            "stable_ridge",
            "teacher_mean",
            "persistence",
        }:
            raise ValueError(
                "fallback_strategy must be one of: "
                "'stable_ridge', 'teacher_mean', 'persistence'."
            )

        if not 0.5 <= self.conformal_coverage <= 0.999:
            raise ValueError("conformal_coverage must be between 0.5 and 0.999.")

    def _validate_last_value_index(self, n_features: int) -> None:
        if self.last_value_index is None:
            return

        idx = int(self.last_value_index)

        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"last_value_index={idx} is out of range for X with "
                f"{n_features} features."
            )

    def _target(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.stationarize:
            return y

        if self.last_value_index is None:
            raise ValueError("last_value_index is required when stationarize=True.")

        return y - X[:, int(self.last_value_index)]

    def _reconstruct(self, X: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if not self.stationarize:
            return pred

        return pred + X[:, int(self.last_value_index)]

    def _context(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d_float64(X, name="X")

        if self.last_value_index is None:
            width = X.shape[1]
        else:
            width = min(X.shape[1], int(self.last_value_index) + 1)

        width = max(1, width)

        return X[:, :width].reshape(X.shape[0], width, 1).astype(np.float32)

    def _chronological_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 20:
            raise ValueError("At least 20 samples are required for reliable calibration.")

        frac = float(np.clip(self.calibration_fraction, 0.05, 0.40))
        cut = int(np.floor(n * (1.0 - frac)))
        cut = min(max(cut, 5), n - 5)

        idx = np.arange(n)

        return idx[:cut], idx[cut:]

    def _normalize_pool(self, pool_preds: np.ndarray) -> np.ndarray:
        if self.pool_mean_ is None or self.pool_std_ is None:
            raise RuntimeError("Pool normalization is not fitted.")

        return (pool_preds - self.pool_mean_) / self.pool_std_

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MQCeNNRegressor":
        self._validate_hyperparameters()
        self._resolve_backend()
        set_global_seed(self.random_state)

        X = _as_2d_float64(X, name="X")
        y = _as_1d_float64(y, name="y")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")

        self._validate_last_value_index(X.shape[1])

        y_target = self._target(X, y)
        train_idx, cal_idx = self._chronological_split(X.shape[0])

        X_train, y_train = X[train_idx], y_target[train_idx]
        X_cal, y_cal = X[cal_idx], y_target[cal_idx]

        self.pool_ = MultiKernelExpertPool(
            kernel_specs=self.kernel_specs,
            n_experts_per_kernel=self.n_experts_per_kernel,
            n_features_per_expert=self.n_features_per_expert,
            alpha_grid=self.alpha_grid,
            random_state=self.random_state,
        ).fit(X_train, y_train)

        pool_train = self.pool_.predict_pool(X_train)
        pool_cal = self.pool_.predict_pool(X_cal)

        self.pool_mean_ = pool_train.mean(axis=0, keepdims=True)
        self.pool_std_ = _safe_std(pool_train, axis=0, keepdims=True)

        fallback_spec = KernelSpec("gaussian", gamma=1.0)
        self.fallback_ = KernelRidgeExpert(
            spec=fallback_spec,
            n_features=max(64, min(512, self.n_features_per_expert)),
            alpha=1.0,
            random_state=self.random_state + 12345,
        ).fit(X_train, y_train)

        self.reliability_ = ReliabilityCalibrator(
            disagreement_weight=1.0,
            novelty_weight=self.novelty_weight,
            sensitivity=self.reliability_sensitivity,
        ).fit(X_train, pool_train)

        self._fit_gate(
            X_train=X_train,
            y_train=y_train,
            pool_train=pool_train,
            X_cal=X_cal,
            y_cal=y_cal,
            pool_cal=pool_cal,
        )

        cal_pred_core = self._predict_core(X_cal, pool_cal)
        cal_reliability = self.reliability_.score(X_cal, pool_cal)

        cal_pred = self._apply_fallback(
            X=X_cal,
            pool_preds=pool_cal,
            pred_core=cal_pred_core,
            reliability=cal_reliability,
        )

        residuals_cal = y_cal - cal_pred

        self.conformal_abs_radius_ = _quantile_abs_residual(
            residuals_cal,
            coverage=self.conformal_coverage,
        )

        fallback_rate_cal = float(
            (cal_reliability < self.reliability_threshold).mean()
        )

        assert self.backend_info_ is not None
        assert self.pool_ is not None

        self.trace_ = MQCeNNTrace(
            best_val_loss=float(self._best_val_loss_),
            epochs_ran=int(self._epochs_ran_),
            n_experts=self.pool_.n_experts_,
            kernel_families=[spec.name for spec in self.kernel_specs],
            best_alpha_per_kernel=dict(self.pool_.best_alpha_),
            reliability_threshold=float(self.reliability_threshold),
            mean_reliability_cal=float(cal_reliability.mean()),
            fallback_rate_cal=fallback_rate_cal,
            conformal_coverage=float(self.conformal_coverage),
            conformal_abs_radius=float(self.conformal_abs_radius_),
            trained_on_samples=int(len(train_idx)),
            calibrated_on_samples=int(len(cal_idx)),
            backend=self.backend_info_.backend,
            device=self.backend_info_.device,
            claim_ledger={
                "quantum_computation": (
                    "No QPU, no quantum circuit, no state-vector simulation."
                ),
                "qml_strengths": (
                    "Implemented as classical, testable proxies."
                ),
                "interference": (
                    "Signed L1 expert weights; validate through softmax ablation."
                ),
                "entanglement": (
                    "Cross-expert interaction bridge; validate through bridge-off ablation."
                ),
                "fail_safety": (
                    "Reliability + fallback + conformal interval; "
                    "not a guarantee of correctness."
                ),
            },
        )

        return self

    def _fit_gate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        pool_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        pool_cal: np.ndarray,
    ) -> None:
        dev = self._device()
        n_experts = pool_train.shape[1]

        X_ctx_train = self._context(X_train)
        X_ctx_cal = self._context(X_cal)

        p_train_norm = self._normalize_pool(pool_train).astype(np.float32)
        p_cal_norm = self._normalize_pool(pool_cal).astype(np.float32)

        x_train_t = torch.as_tensor(X_ctx_train, dtype=torch.float32, device=dev)
        p_train_t = torch.as_tensor(pool_train.astype(np.float32), dtype=torch.float32, device=dev)
        pn_train_t = torch.as_tensor(p_train_norm, dtype=torch.float32, device=dev)
        y_train_t = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32, device=dev)

        x_cal_t = torch.as_tensor(X_ctx_cal, dtype=torch.float32, device=dev)
        p_cal_t = torch.as_tensor(pool_cal.astype(np.float32), dtype=torch.float32, device=dev)
        pn_cal_t = torch.as_tensor(p_cal_norm, dtype=torch.float32, device=dev)
        y_cal_t = torch.as_tensor(y_cal.astype(np.float32), dtype=torch.float32, device=dev)

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

        best_bridge = None
        best_gate = None
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
                ids_np = order[start:start + self.batch_size]
                ids = torch.as_tensor(ids_np, dtype=torch.long, device=dev)

                optimizer.zero_grad()

                bridge = self.bridge_(pn_train_t[ids])
                weights = self.gate_(x_train_t[ids], bridge)
                pred = (weights * p_train_t[ids]).sum(dim=1)

                loss = loss_fn(pred, y_train_t[ids])
                loss.backward()

                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            self.bridge_.eval()
            self.gate_.eval()

            with torch.no_grad():
                bridge_cal = self.bridge_(pn_cal_t)
                weights_cal = self.gate_(x_cal_t, bridge_cal)
                pred_cal = (weights_cal * p_cal_t).sum(dim=1)
                val_loss = float(loss_fn(pred_cal, y_cal_t).detach().cpu())

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_bridge = {
                    k: v.detach().cpu().clone()
                    for k, v in self.bridge_.state_dict().items()
                }
                best_gate = {
                    k: v.detach().cpu().clone()
                    for k, v in self.gate_.state_dict().items()
                }
                no_gain = 0
            else:
                no_gain += 1

                if no_gain >= self.patience:
                    break

        if best_bridge is not None:
            self.bridge_.load_state_dict(best_bridge)

        if best_gate is not None:
            self.gate_.load_state_dict(best_gate)

        if not np.isfinite(best_val):
            self.bridge_.eval()
            self.gate_.eval()

            with torch.no_grad():
                bridge_cal = self.bridge_(pn_cal_t)
                weights_cal = self.gate_(x_cal_t, bridge_cal)
                pred_cal = (weights_cal * p_cal_t).sum(dim=1)
                best_val = float(loss_fn(pred_cal, y_cal_t).detach().cpu())

        self._best_val_loss_ = float(best_val)
        self._epochs_ran_ = int(max(0, epoch + 1))

    def _check_fitted(self) -> None:
        if (
            self.pool_ is None
            or self.fallback_ is None
            or self.bridge_ is None
            or self.gate_ is None
            or self.reliability_ is None
        ):
            raise RuntimeError("The model must be fitted before prediction.")

    def _predict_core(self, X: np.ndarray, pool_preds: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X_ctx = self._context(X)
        pool_norm = self._normalize_pool(pool_preds).astype(np.float32)

        dev = self._device()

        assert self.bridge_ is not None
        assert self.gate_ is not None

        self.bridge_.eval()
        self.gate_.eval()

        with torch.no_grad():
            x_t = torch.as_tensor(X_ctx, dtype=torch.float32, device=dev)
            p_t = torch.as_tensor(pool_preds.astype(np.float32), dtype=torch.float32, device=dev)
            pn_t = torch.as_tensor(pool_norm, dtype=torch.float32, device=dev)

            bridge = self.bridge_(pn_t)
            weights = self.gate_(x_t, bridge)
            pred = (weights * p_t).sum(dim=1).detach().cpu().numpy().astype(np.float64)

        return pred

    def _fallback_prediction(
        self,
        X: np.ndarray,
        pool_preds: np.ndarray,
    ) -> np.ndarray:
        if self.fallback_strategy == "teacher_mean":
            return pool_preds.mean(axis=1)

        if self.fallback_strategy == "persistence":
            if self.last_value_index is None:
                return pool_preds.mean(axis=1)

            if self.stationarize:
                return np.zeros(X.shape[0], dtype=np.float64)

            return X[:, int(self.last_value_index)]

        assert self.fallback_ is not None

        return self.fallback_.predict(X)

    def _apply_fallback(
        self,
        X: np.ndarray,
        pool_preds: np.ndarray,
        pred_core: np.ndarray,
        reliability: np.ndarray,
    ) -> np.ndarray:
        pred = pred_core.copy()
        mask = reliability < self.reliability_threshold

        if np.any(mask):
            pred[mask] = self._fallback_prediction(X[mask], pool_preds[mask])

        return pred

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()

        X = _as_2d_float64(X, name="X")
        self._validate_last_value_index(X.shape[1])

        assert self.pool_ is not None
        assert self.reliability_ is not None

        pool_preds = self.pool_.predict_pool(X)
        reliability = self.reliability_.score(X, pool_preds)

        pred_core = self._predict_core(X, pool_preds)
        pred = self._apply_fallback(X, pool_preds, pred_core, reliability)

        return self._reconstruct(X, pred)

    def predict_interval(
        self,
        X: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return prediction, lower bound and upper bound using split-conformal radius.
        """
        pred = self.predict(X)
        radius = float(self.conformal_abs_radius_)

        if not np.isfinite(radius):
            lower = np.full_like(pred, np.nan, dtype=np.float64)
            upper = np.full_like(pred, np.nan, dtype=np.float64)
        else:
            lower = pred - radius
            upper = pred + radius

        return pred, lower, upper

    def predict_with_diagnostics(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Return prediction plus diagnostics.
        """
        self._check_fitted()

        X = _as_2d_float64(X, name="X")
        self._validate_last_value_index(X.shape[1])

        assert self.pool_ is not None
        assert self.reliability_ is not None

        pool_preds = self.pool_.predict_pool(X)
        reliability = self.reliability_.score(X, pool_preds)

        pred_core = self._predict_core(X, pool_preds)
        pred_target = self._apply_fallback(X, pool_preds, pred_core, reliability)
        pred = self._reconstruct(X, pred_target)

        fallback_mask = reliability < self.reliability_threshold
        radius = float(self.conformal_abs_radius_)

        if np.isfinite(radius):
            interval_lower = pred - radius
            interval_upper = pred + radius
        else:
            interval_lower = np.full_like(pred, np.nan, dtype=np.float64)
            interval_upper = np.full_like(pred, np.nan, dtype=np.float64)

        return {
            "prediction": pred,
            "core_prediction": self._reconstruct(X, pred_core),
            "reliability": reliability,
            "fallback_mask": fallback_mask.astype(bool),
            "interval_lower": interval_lower,
            "interval_upper": interval_upper,
            "teacher_mean": self._reconstruct(X, pool_preds.mean(axis=1)),
            "pool_predictions": pool_preds,
        }

    def predict_teacher_mean(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()

        X = _as_2d_float64(X, name="X")

        assert self.pool_ is not None

        pool_preds = self.pool_.predict_pool(X)

        return self._reconstruct(X, pool_preds.mean(axis=1))


__all__ = [
    "FallbackStrategy",
    "MQCeNNTrace",
    "MQCeNNRegressor",
]
