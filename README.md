# MQ-CeNN: Multi-Quantum Cellular Neural Network

Official Python implementation of **MQ-CeNN**: a claim-bounded, quantum-inspired Cellular Neural Network framework for advanced time-series forecasting.

MQ-CeNN is a fully classical machine-learning framework. It does not implement physical quantum computation, quantum circuits, or quantum state-vector simulation. Instead, it operationalizes selected QML-inspired strengths as testable classical mechanisms for non-stationary time-series forecasting.

## Key Features

* **Heterogeneous kernel expert pool**
  MQ-CeNN builds multiple random-feature ridge experts using different kernel families, including Gaussian, Matérn-like, Laplacian, periodic, and polynomial feature maps.

* **Hilbert-space inspired representation**
  Random Fourier Features and random kitchen-sink projections are used to approximate high-dimensional functional representation spaces.

* **Cellular Neural Gating Network**
  A localized 1D convolutional CeNN-style gate learns how to combine expert predictions from temporal context.

* **Signed interference-style gating**
  MQ-CeNN can use signed L1-normalized expert weights, allowing constructive and destructive expert interaction instead of standard softmax-only voting.

* **Reliability-aware forecasting**
  The framework estimates prediction reliability using expert disagreement and input novelty.

* **Fail-aware fallback mechanism**
  When reliability is low, MQ-CeNN can fall back to a stable ridge expert, teacher mean, or persistence baseline instead of failing silently.

* **Conformal prediction intervals**
  Split-conformal residual calibration provides uncertainty bands around forecasts.

* **Benchmark-ready API**
  The estimator follows the scikit-learn interface with `fit`, `predict`, `predict_interval`, and `predict_with_diagnostics`.

## Installation

Install MQ-CeNN directly from GitHub:

```bash
pip install "git+https://github.com/gedeonnkishi/mq-cenn.git@main"
```

For Kaggle or Google Colab:

```python
!pip install --no-cache-dir "git+https://github.com/gedeonnkishi/mq-cenn.git@main"
```

To force a clean reinstall:

```python
!pip uninstall -y mq-cenn mq_cenn
!pip install --no-cache-dir "git+https://github.com/gedeonnkishi/mq-cenn.git@main"
```

## Quick Smoke Test

```python
import numpy as np
from mq_cenn import MQCeNNRegressor

rng = np.random.default_rng(42)

n = 120
lookback = 24
series = np.sin(np.arange(n) / 6.0) + 0.05 * rng.normal(size=n)

X = []
y = []

for i in range(lookback, n):
    X.append(series[i - lookback:i])
    y.append(series[i])

X = np.asarray(X)
y = np.asarray(y)

model = MQCeNNRegressor(
    n_features_per_expert=32,
    n_experts_per_kernel=1,
    cenn_hidden=8,
    bridge_dim=8,
    cenn_epochs=1,
    batch_size=16,
    patience=1,
    stationarize=True,
    last_value_index=lookback - 1,
    random_state=42,
    device="cpu",
)

model.fit(X[:-10], y[:-10])
pred = model.predict(X[-10:])

print(pred.shape)
print(model.trace_)
```

Expected result:

```text
(10,)
MQCeNNTrace(...)
```

## Basic Usage

```python
from mq_cenn import MQCeNNRegressor

model = MQCeNNRegressor(
    n_features_per_expert=512,
    n_experts_per_kernel=2,
    cenn_epochs=40,
    reliability_threshold=0.30,
    stationarize=True,
    last_value_index=23,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Prediction With Diagnostics

```python
diagnostics = model.predict_with_diagnostics(X_test)

y_pred = diagnostics["prediction"]
reliability = diagnostics["reliability"]
fallback_mask = diagnostics["fallback_mask"]
lower = diagnostics["interval_lower"]
upper = diagnostics["interval_upper"]
```

## Prediction Intervals

```python
pred, lower, upper = model.predict_interval(X_test)
```

## Ablation Suite

```python
from mq_cenn import make_ablation_suite

models = make_ablation_suite({
    "n_features_per_expert": 128,
    "n_experts_per_kernel": 1,
    "cenn_epochs": 10,
    "random_state": 42,
})
```

The ablation suite includes:

* full MQ-CeNN;
* softmax gate variant;
* no periodic kernel variant;
* Gaussian-only variant;
* strict reliability variant;
* no-fallback variant.

## Scientific Claim Discipline

MQ-CeNN is quantum-inspired, not quantum-computational.

The framework does not claim:

* quantum advantage;
* execution on QPU hardware;
* physical superposition;
* physical entanglement;
* quantum tunneling;
* quantum state-vector simulation.

The QML terminology is used only as an operational analogy for classical, testable mechanisms:

* multi-hypothesis representation;
* kernel/Hilbert-space lifting;
* cross-expert interaction;
* signed interference-style weighting;
* reliability-aware prediction;
* fail-aware fallback.

## Recommended Import

```python
from mq_cenn import MQCeNNRegressor
```

The repository name is `mq-cenn`, but the Python module name is `mq_cenn`.

Do not import:

```python
from mq_cenn_candidate import MQCeNNRegressor
from mq_cenn_nmi_candidate import MQCeNNRegressor
```

## License

MIT License.
