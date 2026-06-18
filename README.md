# MQ-CeNN

**Multi-Kernel Quantum-Inspired Cellular Neural Network**

MQ-CeNN is a classical, quantum-inspired machine learning framework for time-series forecasting and anomaly detection. It implements three estimators — single-step regression, direct multi-step regression, and unsupervised anomaly detection — sharing a common architectural core.

The framework does **not** implement physical quantum computation. It does not require a quantum processor, a quantum circuit simulator, or state-vector arithmetic. All mechanisms are implemented as classical, testable proxies that operationalize quantum-inspired design principles on standard hardware.

---

## Table of Contents

- [Scientific Positioning](#scientific-positioning)
- [Architecture](#architecture)
- [Estimators](#estimators)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Ablation Suite](#ablation-suite)
- [Backend Configuration](#backend-configuration)
- [Claim Ledger](#claim-ledger)
- [License](#license)

---

## Scientific Positioning

MQ-CeNN addresses a concrete problem in time-series forecasting: real-world series combine multiple simultaneous structures — periodicity, local trends, regime shifts, noise — that a single model with a fixed inductive bias does not capture uniformly well. The standard response is ensembling, but naive ensembles assume uniform expert utility across all regimes. This assumption does not hold: a periodic expert is useful during cyclic behavior and harmful during structural breaks; a smooth expert is appropriate on stable trends and misleading after a shock.

MQ-CeNN proposes an architecture for **adaptive, heterogeneous expert combination under non-stationarity**, with explicit quantification of prediction reliability.

The quantum-inspired framing maps to concrete classical mechanisms as follows:

| QML concept | Classical implementation in MQ-CeNN |
|---|---|
| Kernel lifting into Hilbert space | Multi-kernel random Fourier feature projection |
| Superposition of hypotheses | Heterogeneous expert pool (5 kernel families) |
| Cross-qubit entanglement | Pairwise interaction bridge (`CrossExpertBridge`) |
| Measurement-induced interference | Signed L1-normalized gate (`SignedInterferenceGate`) |
| Measurement collapse / reliability | Calibrated reliability score (`ReliabilityCalibrator`) |
| Fail-safe readout | Fallback prediction + split-conformal intervals |

Each mechanism is ablatable and tested independently.

---

## Architecture

The core pipeline consists of five components applied in sequence.

### 1. Multi-Kernel Expert Pool (`MultiKernelExpertPool`)

A heterogeneous pool of random-feature ridge regressors. Each expert uses a distinct spectral kernel family:

- **Gaussian** — random Fourier features with Gaussian spectral measure; appropriate for smooth nonlinear structure.
- **Matérn-3/2** — heavy-tailed spectral proxy; captures less smooth, locally irregular behavior.
- **Laplacian** — Cauchy spectral proxy; more robust to large input deviations.
- **Periodic** — harmonic random-kitchen-sink features; captures cyclic patterns with known or unknown period.
- **Polynomial** — bounded polynomial random projection; captures interactions and trend components.

Each family contributes `n_experts_per_kernel` experts with independently drawn random weights, producing a pool of `K = 5 × n_experts_per_kernel` regressors. Ridge regularization strength is selected per family via chronological `TimeSeriesSplit` cross-validation.

For multi-step regression, each expert solves a multi-output ridge problem with target matrix `Y ∈ ℝ^(n × H)` where `H` is the forecast horizon.

### 2. Cross-Expert Bridge (`CrossExpertBridge`)

A two-layer MLP that receives the normalized pool predictions and enriches them with pairwise expert interactions before gating:

```
outer_ij = p̂_i · p̂_j       (outer product, K² terms)
z = [p̂; outer]              (K + K² features)
bridge = MLP(z)              (output: ℝ^bridge_dim)
```

This is a classical proxy for cross-expert dependence. It allows the gate to condition its weights on correlations and disagreements between experts, not only on the raw predictions.

### 3. Signed Interference Gate (`SignedInterferenceGate`)

A temporal convolutional encoder that processes the input window and produces expert weights:

```
h = Conv1D-Tanh-Conv1D-Tanh(x_seq)     (temporal encoding)
logits = Linear([h_mean; bridge])        (combine temporal + bridge)
weights = tanh(logits) / ||tanh(logits)||_1    (signed L1 normalization)
```

The signed normalization allows negative weights, meaning an expert can subtract its contribution if it introduces a redundant or biased signal. This differs from softmax ensembling, which constrains all weights to be positive and sum to one. The softmax variant is available as an ablation baseline (`signed_interference=False`).

For multi-step regression, the gate produces one weight per expert (scalar), shared across all horizon steps. This regularizes the selection toward experts that are globally coherent over the full prediction horizon.

### 4. Reliability Calibrator (`ReliabilityCalibrator`)

A post-hoc reliability score in [0, 1] computed from two orthogonal signals:

```
disagreement(x) = Var_k[f_k(x)] / disagreement_ref
novelty(x)      = d_Mahalanobis(x, X_train) / novelty_ref
energy(x)       = w_d · disagreement(x) + w_n · novelty(x)
reliability(x)  = exp(−sensitivity · energy(x))
```

The Mahalanobis distance uses median and median absolute deviation estimates for robustness. A reliability score close to 1 indicates an input well within the training distribution with high expert consensus. A score close to 0 indicates either a novel input or strong expert disagreement, triggering fallback.

For the `MQCeNNAnomalyDetector`, the reliability score is negated to produce an anomaly score following the sklearn `OutlierMixin` convention: more negative = more anomalous.

### 5. Fallback and Conformal Intervals

When `reliability(x) < threshold`:

- **`stable_ridge`** (default, single-step only): a pre-fitted Gaussian ridge expert trained on the same data.
- **`teacher_mean`**: unweighted average of the expert pool.
- **`persistence`**: last observed value (or zero in stationarized mode).

Prediction intervals use split-conformal calibration: the interval radius is the empirical quantile of absolute residuals on the held-out calibration fold at the requested coverage level.

---

## Estimators

### `MQCeNNRegressor` — single-step forecasting

Predicts `y_{t+1}` from a lookback window `x_t = [x_{t-W}, ..., x_{t-1}]`. Targets are scalar. The full pipeline (pool → bridge → gate → reliability → fallback → conformal) is active.

**Input format:**
- `X`: `(n_samples, window_size)` — flattened lookback windows.
- `y`: `(n_samples,)` — scalar targets.

**Key outputs:**
- `predict(X)` → `(n_samples,)`
- `predict_interval(X)` → `(prediction, lower, upper)`, each `(n_samples,)`
- `predict_with_diagnostics(X)` → dict with `prediction`, `reliability`, `fallback_mask`, `pool_predictions`, `interval_lower`, `interval_upper`.

---

### `MQCeNNMultiStepRegressor` — direct multi-step forecasting

Predicts the full horizon `[y_{t+1}, ..., y_{t+H}]` simultaneously from a single lookback window. Avoids the error accumulation of recursive single-step strategies.

Architectural differences from `MQCeNNRegressor`:

- Each expert solves multi-output ridge: `β ∈ ℝ^(D_rff × H)`, one Cholesky factorization per expert rather than one per horizon step.
- Pool predictions have shape `(n_samples, n_experts, H)`.
- The gate weight vector `w ∈ ℝ^K` is shared across all horizon steps: `ŷ = Σ_k w_k · f_k(x)` where `f_k(x) ∈ ℝ^H`.
- Reliability is computed on the mean collapsed expert prediction `mean_H(pool_preds)`.

**Input format:**
- `X`: `(n_samples, window_size)` — produced by `make_multistep_windows`.
- `Y`: `(n_samples, horizon)` — multi-step target matrix.

**Key output:** `predict(X)` → `(n_samples, horizon)`.

Natural benchmark horizons: H ∈ {24, 48, 96, 192, 336, 720} on ETTh1, ETTm1/2, Weather, Traffic.

---

### `MQCeNNAnomalyDetector` — unsupervised anomaly detection

Trains on data assumed to be normal. No anomaly labels are used. Anomaly scores are derived directly from the `ReliabilityCalibrator`.

The pool is fitted using a self-supervised reconstruction proxy: each window predicts its own center value (`y_proxy = X[:, W//2]`). This proxy is not predictively accurate; its role is to induce expert diversity and disagreement on abnormal inputs. The `ReliabilityCalibrator` then measures both this disagreement and the input novelty relative to training.

The threshold can be fixed manually (`reliability_threshold`) or auto-calibrated from the expected contamination rate of training data (`anomaly_rate_train`).

**Input format:**
- `fit(X)`: `(n_samples, window_size)` from a reference period assumed normal.
- `score_samples(X)` → `(n_samples,)`, convention: more negative = more anomalous.
- `predict(X)` → `(n_samples,)` with labels `+1` (normal), `-1` (anomaly).
- `reliability_scores(X)` → `(n_samples,)` in [0, 1], for plotting and manual thresholding.

---

## Installation

### Standard

```bash
pip install -e .
```

### Development

```bash
pip install -e .[dev]
# or
pip install -r requirements-dev.txt
```

### Dependencies

| Package | Minimum version |
|---|---|
| `numpy` | 1.22 |
| `torch` | 2.0 |
| `scikit-learn` | 1.0 |
| `scipy` | 1.8 |
| `pandas` | 1.4 |
| `matplotlib` | 3.5 |

---

## Quickstart

### Single-step forecasting

```python
import numpy as np
from mq_cenn import (
    MQCeNNRegressor,
    make_supervised_windows,
    chronological_split,
    train_only_standardize,
)

series = np.sin(np.linspace(0, 80, 1200)) + 0.05 * np.random.randn(1200)

X, y = make_supervised_windows(series, lookback=96, horizon=1, flatten=True)
X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)
X_train, X_val, X_test = train_only_standardize(X_train, X_val, X_test)

model = MQCeNNRegressor(
    n_features_per_expert=512,
    n_experts_per_kernel=2,
    cenn_epochs=40,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
pred, lower, upper = model.predict_interval(X_test)
diag = model.predict_with_diagnostics(X_test)
```

### Multi-step forecasting

```python
from mq_cenn import MQCeNNMultiStepRegressor, make_multistep_windows

X, Y = make_multistep_windows(series, lookback=96, horizon=24, flatten=True)
X_train, Y_train, X_val, Y_val, X_test, Y_test = chronological_split(X, Y)
X_train, X_val, X_test = train_only_standardize(X_train, X_val, X_test)

model = MQCeNNMultiStepRegressor(
    horizon=24,
    n_features_per_expert=512,
    n_experts_per_kernel=2,
    cenn_epochs=40,
    random_state=42,
)
model.fit(X_train, Y_train)

preds = model.predict(X_test)    # shape (n_test, 24)
diag  = model.predict_with_diagnostics(X_test)
```

### Anomaly detection

```python
from mq_cenn import MQCeNNAnomalyDetector, make_supervised_windows

X_normal, _ = make_supervised_windows(normal_series, lookback=48, horizon=1, flatten=True)
X_normal = train_only_standardize(X_normal)

detector = MQCeNNAnomalyDetector(
    n_features_per_expert=256,
    novelty_weight=0.5,
    anomaly_rate_train=0.02,    # auto-calibrate threshold; 0 to use reliability_threshold directly
    random_state=42,
)
detector.fit(X_normal)

scores = detector.score_samples(X_test)     # (n,) — more negative = more anomalous
labels = detector.predict(X_test)           # (n,) — +1 normal, -1 anomaly
rel    = detector.reliability_scores(X_test) # (n,) in [0, 1] — for plotting
```

---

## API Reference

### Public symbols

```python
from mq_cenn import (
    # Estimators
    MQCeNNRegressor,
    MQCeNNMultiStepRegressor,
    MQCeNNAnomalyDetector,

    # Trace dataclasses
    MQCeNNTrace,
    MQCeNNMultiStepTrace,
    MQCeNNAnomalyTrace,

    # Core components (for custom extensions)
    KernelSpec,
    KernelName,
    DEFAULT_KERNEL_SPECS,
    SpectralFeatureProjector,
    KernelRidgeExpert,
    MultiKernelExpertPool,
    CrossExpertBridge,
    SignedInterferenceGate,
    ReliabilityCalibrator,
    NoveltyDetector,

    # Preprocessing
    make_supervised_windows,
    make_multistep_windows,
    chronological_split,
    train_only_standardize,
    flatten_windows,
    make_calendar_features,
    add_calendar_features,
    add_seasonal_lag_features,
    make_seasonal_lag_matrix,
    seasonal_naive_forecast,

    # Ablation
    make_ablation_suite,

    # Utilities
    set_global_seed,
)
```

### `MQCeNNRegressor` — key parameters

| Parameter | Default | Description |
|---|---|---|
| `n_features_per_expert` | 512 | Random Fourier features per expert |
| `n_experts_per_kernel` | 2 | Experts per kernel family |
| `kernel_specs` | DEFAULT_KERNEL_SPECS | Kernel families (5 defaults) |
| `alpha_grid` | (1e-3 … 10) | Ridge alpha grid for TimeSeriesSplit selection |
| `bridge_dim` | 32 | CrossExpertBridge output dimension |
| `cenn_hidden` | 64 | Gate temporal encoder hidden size |
| `cenn_kernel` | 3 | Gate Conv1D kernel size |
| `cenn_epochs` | 40 | Maximum training epochs |
| `cenn_lr` | 1e-3 | AdamW learning rate |
| `patience` | 6 | Early stopping patience |
| `calibration_fraction` | 0.15 | Chronological calibration split |
| `signed_interference` | True | Signed gate (False = softmax ablation) |
| `reliability_threshold` | 0.30 | Fallback trigger threshold |
| `fallback_strategy` | `"stable_ridge"` | `"stable_ridge"` / `"teacher_mean"` / `"persistence"` |
| `conformal_coverage` | 0.90 | Target coverage for prediction intervals |
| `stationarize` | False | First-difference target before fitting |
| `random_state` | 42 | Global seed |

### `MQCeNNMultiStepRegressor` — additional parameters

| Parameter | Default | Description |
|---|---|---|
| `horizon` | 24 | Forecast horizon H |
| `alpha` | 1.0 | Fixed ridge penalty (no grid search in multi-step mode) |
| `fallback_strategy` | `"teacher_mean"` | `"teacher_mean"` or `"persistence"` (`"stable_ridge"` not supported) |

### `MQCeNNAnomalyDetector` — key parameters

| Parameter | Default | Description |
|---|---|---|
| `novelty_weight` | 0.5 | Weight of input novelty in reliability score |
| `reliability_sensitivity` | 1.0 | Exponential decay rate |
| `reliability_threshold` | 0.30 | Anomaly threshold when `anomaly_rate_train=0` |
| `anomaly_rate_train` | 0.0 | Auto-calibrate threshold to flag this fraction of training data |

### Windowing utilities

```python
# Single-step: y is scalar (target at t+horizon)
X, y = make_supervised_windows(series, lookback=96, horizon=1, flatten=True)
# X: (n, 96)  y: (n,)

# Multi-step: Y is the full horizon vector [t+1, ..., t+H]
X, Y = make_multistep_windows(series, lookback=96, horizon=24, flatten=True)
# X: (n, 96)  Y: (n, 24)

# Three-way chronological split (no leakage)
X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
    X, y, train_frac=0.70, val_frac=0.15
)

# Train-only standardization (prevents leakage from test statistics)
X_train, X_val, X_test = train_only_standardize(X_train, X_val, X_test)
```

---

## Ablation Suite

The ablation suite produces six named estimator variants for controlled comparison:

```python
from mq_cenn import make_ablation_suite

suite = make_ablation_suite(
    n_features_per_expert=512,
    cenn_epochs=40,
    random_state=42,
)

for name, estimator in suite.items():
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
```

| Variant | What is disabled |
|---|---|
| `MQCeNN_full` | Nothing — full model |
| `MQCeNN_softmax_gate` | Signed interference → softmax weighting |
| `MQCeNN_no_periodic_kernel` | Periodic kernel family |
| `MQCeNN_gaussian_only` | All families except Gaussian |
| `MQCeNN_strict_reliability` | Stricter fallback trigger |
| `MQCeNN_no_fallback` | Fallback disabled |

Each variant must be evaluated on the same data split, with the same seed and metrics. The suite does not select the best variant automatically.

---

## Backend Configuration

MQ-CeNN defaults to NumPy/CPU. The backend dispatcher auto-detects available hardware:

```python
from mq_cenn.backends import resolve_backend

info = resolve_backend("auto", "auto")
print(info)
# BackendInfo(backend='numpy', device='cpu', cuda_available=False, ...)
```

Available backends:

| Backend | Requirement |
|---|---|
| `numpy` (default) | NumPy ≥ 1.22 |
| `cpp` | C++17 compiler, pybind11; build with `MQCENN_BUILD_CPP=1` |
| `cuda` | NVIDIA GPU, CUDA Toolkit, CUDA-compatible PyTorch; build with `MQCENN_BUILD_CUDA=1` |

CLI environment check:

```bash
mqcenn doctor
mqcenn install-backend --cpu     # NumPy (default)
mqcenn install-backend --cpp     # C++ extension
mqcenn install-backend --cuda    # CUDA extension
```

The neural components (bridge, gate) always run through PyTorch and respect the `device` parameter independently of the backend used for the expert pool.

---

## Claim Ledger

Every fitted MQ-CeNN estimator exposes a `trace_.claim_ledger` dictionary that explicitly documents the scientific status of each mechanism:

```python
model.fit(X_train, y_train)
for claim, status in model.trace_.claim_ledger.items():
    print(f"{claim}: {status}")
```

Documented claims for `MQCeNNRegressor`:

| Claim | Status |
|---|---|
| `quantum_computation` | No QPU, no quantum circuit, no state-vector simulation |
| `interference` | Signed L1 expert weights; validate through softmax ablation |
| `entanglement` | Cross-expert interaction bridge; validate through bridge-off ablation |
| `fail_safety` | Reliability + fallback + conformal interval; not a correctness guarantee |

---

## Testing

```bash
pip install -e .[dev]
pytest               # full suite (expected: 49 passed)
pytest tests/test_regressor.py   # single file
```

---

## License

Released under the MIT License. See `LICENSE` for details.
