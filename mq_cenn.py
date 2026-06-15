# =============================================================================
# mq_cenn_nmi_candidate.py
#
# MQ-CeNN — claim-bounded QML-strength emulator for time-series forecasting.
#
# This module does NOT implement physical quantum computation.
# It operationalizes QML-inspired strengths as testable classical mechanisms:
#
#   1) Multi-hypothesis representation
#      -> heterogeneous spectral kernel experts, not only different seeds.
#
#   2) Hilbert-space kernel lifting
#      -> Random Fourier / random kitchen-sink projections with ridge heads.
#
#   3) Cross-expert dependence
#      -> pairwise expert interaction bridge before gating.
#
#   4) Interference-like combination
#      -> signed L1-normalized expert weights, unlike softmax voting.
#
#   5) Measurement / reliability
#      -> calibrated reliability score from expert disagreement + novelty.
#
#   6) Fail-aware prediction
#      -> conformal residual band + fallback when reliability is low.
#
# Scientific claim discipline:
#   - Do not claim quantum advantage.
#   - Do not claim real superposition, entanglement, or tunneling.
#   - Report these as classical operational analogues.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit


ArrayLike = Union[np.ndarray, Sequence[float]]
KernelName = Literal["gaussian", "matern32", "laplacian", "periodic", "polynomial"]
FallbackStrategy = Literal["stable_ridge", "teacher_mean", "persistence"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _as_float64(x: ArrayLike, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_2d_float64(x: ArrayLike, *, name: str = "X") -> np.ndarray:
    arr = _as_float64(x, name=name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


def _as_1d_float64(y: ArrayLike, *, name: str = "y") -> np.ndarray:
    arr = _as_float64(y, name=name).reshape(-1)
    return arr


def _safe_std(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.std(x, axis=axis, keepdims=keepdims) + 1e-8


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _quantile_abs_residual(residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal absolute residual quantile."""
    if residuals.size == 0:
        return float("nan")
    coverage = float(np.clip(coverage, 0.50, 0.999))
    q = np.ceil((residuals.size + 1) * coverage) / residuals.size
    q = min(1.0, q)
    return float(np.quantile(np.abs(residuals), q))


# ---------------------------------------------------------------------------
# Kernel specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelSpec:
    """One expert family.

    The names are intentionally classical. They are QML-inspired only through
    kernel/Hilbert-space lifting and spectral random features.
    """
    name: KernelName
    gamma: float = 1.0
    period: float = 24.0
    degree: int = 2


DEFAULT_KERNEL_SPECS: Tuple[KernelSpec, ...] = (
    KernelSpec("gaussian", gamma=1.0),
    KernelSpec("matern32", gamma=0.8),
    KernelSpec("laplacian", gamma=0.5),
    KernelSpec("periodic", gamma=1.0, period=24.0),
    KernelSpec("polynomial", gamma=1.0, degree=2),
)


# ---------------------------------------------------------------------------
# Random feature projectors
# ---------------------------------------------------------------------------

class SpectralFeatureProjector:
    """Random feature map for a family-specific kernel proxy.

    Gaussian, Matérn and Laplacian use spectral random features.
    Periodic and polynomial are practical random-kitchen-sink proxies.

    This is deliberately claim-bounded: these features are not quantum circuits.
    """

    SUPPORTED: Tuple[str, ...] = ("gaussian", "matern32", "laplacian", "periodic", "polynomial")

    def __init__(
        self,
        spec: KernelSpec,
        n_features: int = 512,
        random_state: int = 42,
    ):
        if spec.name not in self.SUPPORTED:
            raise ValueError(f"Unsupported kernel '{spec.name}'.")
        if n_features < 4:
            raise ValueError("n_features must be >= 4.")
        self.spec = spec
        self.n_features = int(n_features)
        self.random_state = int(random_state)
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None
        self.scale_: float = np.sqrt(2.0 / self.n_features)

    def fit(self, n_input_features: int) -> "SpectralFeatureProjector":
        rng = np.random.default_rng(self.random_state)
        d = int(n_input_features)
        g = max(float(self.spec.gamma), 1e-12)

        if self.spec.name == "gaussian":
            self.W_ = rng.normal(0.0, np.sqrt(2.0 * g), size=(d, self.n_features))
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "matern32":
            # Matérn-like heavy-tailed spectral measure.
            # This is a stable proxy, not an exact parameterization claim.
            nu = 3.0
            z = rng.normal(0.0, 1.0, size=(d, self.n_features))
            v = rng.chisquare(nu, size=(1, self.n_features)) / nu
            self.W_ = np.sqrt(3.0 * g) * z / np.sqrt(v + 1e-12)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "laplacian":
            # Cauchy spectral proxy, clipped for numerical stability.
            u = rng.uniform(1e-6, 1.0 - 1e-6, size=(d, self.n_features))
            self.W_ = g * np.tan(np.pi * (u - 0.5))
            self.W_ = np.clip(self.W_, -50.0, 50.0)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "periodic":
            period = max(float(self.spec.period), 1e-6)
            base = 2.0 * np.pi / period
            harmonics = rng.integers(1, max(2, self.n_features // 2), size=(d, self.n_features))
            signs = rng.choice([-1.0, 1.0], size=(d, self.n_features))
            self.W_ = signs * harmonics * base * np.sqrt(g)
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)

        elif self.spec.name == "polynomial":
            # Random projection polynomial proxy.
            self.W_ = rng.normal(0.0, np.sqrt(g), size=(d, self.n_features))
            self.b_ = rng.normal(0.0, 1.0, size=self.n_features)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Projector must be fitted before transform().")
        X = _as_2d_float64(X, name="X")
        z = X @ self.W_ + self.b_

        if self.spec.name == "polynomial":
            degree = max(1, int(self.spec.degree))
            # Bounded polynomial random kitchen-sink proxy.
            z = np.tanh(z)
            phi = z
            if degree > 1:
                phi = np.sign(z) * (np.abs(z) ** degree)
            return phi / np.sqrt(self.n_features)

        return self.scale_ * np.cos(z)


# ---------------------------------------------------------------------------
# Ridge expert
# ---------------------------------------------------------------------------

class KernelRidgeExpert:
    """One random-feature ridge expert."""

    def __init__(
        self,
        spec: KernelSpec,
        n_features: int,
        alpha: float,
        random_state: int,
    ):
        self.spec = spec
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

        self.projector_: Optional[SpectralFeatureProjector] = None
        self.beta_: Optional[np.ndarray] = None
        self.train_residual_std_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeExpert":
        X = _as_2d_float64(X, name="X")
        y = _as_1d_float64(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")

        self.projector_ = SpectralFeatureProjector(
            spec=self.spec,
            n_features=self.n_features,
            random_state=self.random_state,
        ).fit(X.shape[1])

        Z = self.projector_.transform(X)
        A = Z.T @ Z
        A.flat[:: A.shape[0] + 1] += self.alpha
        rhs = Z.T @ y

        try:
            self.beta_ = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(A, rhs, rcond=None)[0]

        residuals = y - (Z @ self.beta_)
        self.train_residual_std_ = float(np.std(residuals) + 1e-8)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.projector_ is None or self.beta_ is None:
            raise RuntimeError("Expert must be fitted before predict().")
        X = _as_2d_float64(X, name="X")
        return self.projector_.transform(X) @ self.beta_


class MultiKernelExpertPool:
    """Heterogeneous pool: different kernels + different random projections.

    This is the main upgrade over a same-kernel, multi-seed pool.
    """

    def __init__(
        self,
        kernel_specs: Sequence[KernelSpec] = DEFAULT_KERNEL_SPECS,
        n_experts_per_kernel: int = 2,
        n_features_per_expert: int = 512,
        alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        n_splits: int = 3,
        random_state: int = 42,
    ):
        self.kernel_specs = tuple(kernel_specs)
        self.n_experts_per_kernel = int(n_experts_per_kernel)
        self.n_features_per_expert = int(n_features_per_expert)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

        self.experts_: List[KernelRidgeExpert] = []
        self.best_alpha_: Dict[str, float] = {}

    @property
    def n_experts_(self) -> int:
        return len(self.experts_)

    def _select_alpha(self, X: np.ndarray, y: np.ndarray, spec: KernelSpec) -> float:
        if X.shape[0] < max(40, self.n_splits + 5):
            return self.alpha_grid[len(self.alpha_grid) // 2]

        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        best_alpha = self.alpha_grid[0]
        best_loss = np.inf

        for alpha in self.alpha_grid:
            losses = []
            for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(X)):
                expert = KernelRidgeExpert(
                    spec=spec,
                    n_features=self.n_features_per_expert,
                    alpha=alpha,
                    random_state=self.random_state + 1009 * fold_id,
                )
                expert.fit(X[tr_idx], y[tr_idx])
                pred = expert.predict(X[va_idx])
                losses.append(float(np.mean(np.abs(pred - y[va_idx]))))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_loss = loss
                best_alpha = float(alpha)

        return best_alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiKernelExpertPool":
        X = _as_2d_float64(X, name="X")
        y = _as_1d_float64(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch.")

        self.experts_ = []
        self.best_alpha_ = {}
        seed_counter = 0

        for spec in self.kernel_specs:
            alpha = self._select_alpha(X, y, spec)
            self.best_alpha_[spec.name] = alpha
            for _ in range(self.n_experts_per_kernel):
                expert = KernelRidgeExpert(
                    spec=spec,
                    n_features=self.n_features_per_expert,
                    alpha=alpha,
                    random_state=self.random_state + 9973 * seed_counter,
                )
                expert.fit(X, y)
                self.experts_.append(expert)
                seed_counter += 1

        return self

    def predict_pool(self, X: np.ndarray) -> np.ndarray:
        if not self.experts_:
            raise RuntimeError("Expert pool must be fitted before predict_pool().")
        X = _as_2d_float64(X, name="X")
        return np.column_stack([expert.predict(X) for expert in self.experts_])

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.predict_pool(X).mean(axis=1)


# ---------------------------------------------------------------------------
# Reliability and novelty calibration
# ---------------------------------------------------------------------------

class NoveltyDetector:
    """Diagonal Mahalanobis-like novelty score in input space."""

    def __init__(self):
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
        z = (X - self.center_) / self.scale_
        return np.sqrt(np.mean(z * z, axis=1))


class ReliabilityCalibrator:
    """Reliability score from expert disagreement and input novelty.

    r_t close to 1: in-regime, experts agree.
    r_t close to 0: out-of-regime and/or experts disagree.
    """

    def __init__(
        self,
        disagreement_weight: float = 1.0,
        novelty_weight: float = 0.5,
        sensitivity: float = 1.0,
    ):
        self.disagreement_weight = float(disagreement_weight)
        self.novelty_weight = float(novelty_weight)
        self.sensitivity = float(sensitivity)
        self.disagreement_ref_: float = 1.0
        self.novelty_detector_: Optional[NoveltyDetector] = None

    def fit(self, X_ref: np.ndarray, pool_preds_ref: np.ndarray) -> "ReliabilityCalibrator":
        X_ref = _as_2d_float64(X_ref, name="X_ref")
        pool_preds_ref = _as_2d_float64(pool_preds_ref, name="pool_preds_ref")

        disagreement = np.var(pool_preds_ref, axis=1)
        self.disagreement_ref_ = float(np.median(disagreement) + 1e-8)
        self.novelty_detector_ = NoveltyDetector().fit(X_ref)
        return self

    def score(self, X: np.ndarray, pool_preds: np.ndarray) -> np.ndarray:
        if self.novelty_detector_ is None:
            raise RuntimeError("ReliabilityCalibrator must be fitted before score().")
        X = _as_2d_float64(X, name="X")
        pool_preds = _as_2d_float64(pool_preds, name="pool_preds")

        disagreement = np.var(pool_preds, axis=1) / self.disagreement_ref_
        novelty = self.novelty_detector_.score(X) / self.novelty_detector_.ref_

        energy = (
            self.disagreement_weight * disagreement
            + self.novelty_weight * novelty
        )
        return np.exp(-self.sensitivity * energy)


# ---------------------------------------------------------------------------
# Neural bridge and signed gating
# ---------------------------------------------------------------------------

class CrossExpertBridge(nn.Module):
    """Pairwise interaction bridge over expert predictions.

    It is an entanglement proxy only in the weak, operational sense:
    expert outputs are no longer treated as independent coordinates.
    """

    def __init__(self, n_experts: int, bridge_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.n_experts = int(n_experts)
        cross_dim = self.n_experts * self.n_experts

        self.net = nn.Sequential(
            nn.Linear(cross_dim + self.n_experts, bridge_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim, bridge_dim),
            nn.Tanh(),
            nn.LayerNorm(bridge_dim),
        )

    def forward(self, pool_preds_norm: torch.Tensor) -> torch.Tensor:
        outer = torch.bmm(
            pool_preds_norm.unsqueeze(2),
            pool_preds_norm.unsqueeze(1),
        ).reshape(pool_preds_norm.shape[0], -1)
        z = torch.cat([pool_preds_norm, outer], dim=-1)
        return self.net(z)


class SignedInterferenceGate(nn.Module):
    """Temporal gate that can assign positive or negative weights.

    Softmax gates cannot cancel an expert; they can only reduce its mass.
    Signed L1-normalized weights make cancellation testable by ablation.
    """

    def __init__(
        self,
        context_channels: int,
        n_experts: int,
        bridge_dim: int = 32,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.05,
        signed: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        self.signed = bool(signed)
        self.temporal = nn.Sequential(
            nn.Conv1d(context_channels, hidden_dim, kernel_size, padding=padding),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + bridge_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x_seq: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, seq_len, context_channels)
        h = self.temporal(x_seq.permute(0, 2, 1)).mean(dim=-1)
        logits = self.head(torch.cat([h, bridge], dim=-1))

        if not self.signed:
            return torch.softmax(logits, dim=-1)

        raw = torch.tanh(logits)
        denom = raw.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return raw / denom


# ---------------------------------------------------------------------------
# Trace / diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MQCeNNTrace:
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
    claim_ledger: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class MQCeNNRegressor(BaseEstimator, RegressorMixin):
    """MQ-CeNN estimator.

    This estimator is designed for rigorous benchmarking rather than hype:
    it exposes ablations and diagnostics needed to defend or reject the idea.

    Main anti-leakage choice:
    a chronological calibration split is kept out of the expert-pool fit and
    out of neural-gate optimization. It is used only for early stopping,
    reliability diagnostics, and conformal residual calibration.
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
        device: Optional[str] = None,
    ):
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
        self.device = device

        self.pool_: Optional[MultiKernelExpertPool] = None
        self.fallback_: Optional[KernelRidgeExpert] = None
        self.bridge_: Optional[CrossExpertBridge] = None
        self.gate_: Optional[SignedInterferenceGate] = None
        self.reliability_: Optional[ReliabilityCalibrator] = None
        self.trace_: Optional[MQCeNNTrace] = None

        self.pool_mean_: Optional[np.ndarray] = None
        self.pool_std_: Optional[np.ndarray] = None
        self.conformal_abs_radius_: float = float("nan")

    # ------------------------- internal helpers -------------------------

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ------------------------- fit -------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MQCeNNRegressor":
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

        # Experts are fitted only on the chronological training block.
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

        # Stable fallback fitted on the same non-calibration training block.
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

        # Calibration diagnostics and conformal residuals.
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

        fallback_rate_cal = float((cal_reliability < self.reliability_threshold).mean())

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
            claim_ledger={
                "quantum_computation": "No QPU, no quantum circuit, no state-vector simulation.",
                "qml_strengths": "Implemented as classical, testable proxies.",
                "interference": "Signed L1 expert weights; validate through softmax ablation.",
                "entanglement": "Cross-expert interaction bridge; validate through bridge-off ablation.",
                "fail_safety": "Reliability + fallback + conformal interval; not a guarantee of correctness.",
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

        best_bridge, best_gate = None, None
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
                best_bridge = {k: v.detach().cpu().clone() for k, v in self.bridge_.state_dict().items()}
                best_gate = {k: v.detach().cpu().clone() for k, v in self.gate_.state_dict().items()}
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= self.patience:
                    break

        if best_bridge is not None:
            self.bridge_.load_state_dict(best_bridge)
        if best_gate is not None:
            self.gate_.load_state_dict(best_gate)

        self._best_val_loss_ = float(best_val)
        self._epochs_ran_ = int(epoch + 1)

    # ------------------------- prediction internals -------------------------

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

    def _fallback_prediction(self, X: np.ndarray, pool_preds: np.ndarray) -> np.ndarray:
        if self.fallback_strategy == "teacher_mean":
            return pool_preds.mean(axis=1)
        if self.fallback_strategy == "persistence":
            if self.last_value_index is None:
                return pool_preds.mean(axis=1)
            # In target space: if stationarized, persistence increment is 0.
            if self.stationarize:
                return np.zeros(X.shape[0], dtype=np.float64)
            return X[:, int(self.last_value_index)]
        # stable_ridge
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

    # ------------------------- public prediction methods -------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X = _as_2d_float64(X, name="X")
        self._validate_last_value_index(X.shape[1])

        pool_preds = self.pool_.predict_pool(X)
        reliability = self.reliability_.score(X, pool_preds)
        pred_core = self._predict_core(X, pool_preds)
        pred = self._apply_fallback(X, pool_preds, pred_core, reliability)
        return self._reconstruct(X, pred)

    def predict_interval(
        self,
        X: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prediction, lower, upper using split-conformal residual radius."""
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
        """Return predictions plus reliability, fallback mask, intervals and raw pool."""
        self._check_fitted()
        X = _as_2d_float64(X, name="X")
        self._validate_last_value_index(X.shape[1])

        pool_preds = self.pool_.predict_pool(X)
        reliability = self.reliability_.score(X, pool_preds)
        pred_core = self._predict_core(X, pool_preds)
        pred_target = self._apply_fallback(X, pool_preds, pred_core, reliability)
        pred = self._reconstruct(X, pred_target)

        fallback_mask = reliability < self.reliability_threshold
        radius = float(self.conformal_abs_radius_)

        return {
            "prediction": pred,
            "core_prediction": self._reconstruct(X, pred_core),
            "reliability": reliability,
            "fallback_mask": fallback_mask.astype(bool),
            "interval_lower": pred - radius if np.isfinite(radius) else np.full_like(pred, np.nan),
            "interval_upper": pred + radius if np.isfinite(radius) else np.full_like(pred, np.nan),
            "teacher_mean": self._reconstruct(X, pool_preds.mean(axis=1)),
            "pool_predictions": pool_preds,
        }

    def predict_teacher_mean(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        X = _as_2d_float64(X, name="X")
        pool_preds = self.pool_.predict_pool(X)
        return self._reconstruct(X, pool_preds.mean(axis=1))


# ---------------------------------------------------------------------------
# Minimal ablation factory
# ---------------------------------------------------------------------------

def make_ablation_suite(
    base_kwargs: Optional[Dict] = None,
) -> Dict[str, MQCeNNRegressor]:
    """Create ablation models required for a serious paper.

    Use the same data splits and metrics outside this function.
    """
    base_kwargs = dict(base_kwargs or {})

    full = MQCeNNRegressor(**base_kwargs)

    softmax_kwargs = dict(base_kwargs)
    softmax_kwargs["signed_interference"] = False

    no_periodic_kwargs = dict(base_kwargs)
    no_periodic_kwargs["kernel_specs"] = tuple(
        spec for spec in DEFAULT_KERNEL_SPECS if spec.name != "periodic"
    )

    gaussian_only_kwargs = dict(base_kwargs)
    gaussian_only_kwargs["kernel_specs"] = (KernelSpec("gaussian", gamma=1.0),)

    high_threshold_kwargs = dict(base_kwargs)
    high_threshold_kwargs["reliability_threshold"] = 0.60

    no_fallback_kwargs = dict(base_kwargs)
    no_fallback_kwargs["reliability_threshold"] = -1.0

    return {
        "MQCeNN_full": full,
        "MQCeNN_softmax_gate": MQCeNNRegressor(**softmax_kwargs),
        "MQCeNN_no_periodic_kernel": MQCeNNRegressor(**no_periodic_kwargs),
        "MQCeNN_gaussian_only": MQCeNNRegressor(**gaussian_only_kwargs),
        "MQCeNN_strict_reliability": MQCeNNRegressor(**high_threshold_kwargs),
        "MQCeNN_no_fallback": MQCeNNRegressor(**no_fallback_kwargs),
    }


__all__ = [
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "MQCeNNRegressor",
    "MQCeNNTrace",
    "make_ablation_suite",
    "set_global_seed",
]
