# MQ-CeNN

**MQ-CeNN** is a classical, quantum-inspired machine learning framework for time-series forecasting and regression.

The framework combines:

- multi-kernel random-feature experts;
- cross-expert interaction bridge;
- signed CeNN-inspired temporal interference gate;
- reliability calibration;
- fallback prediction;
- split-conformal prediction intervals;
- automatic backend detection for CPU, C++ and CUDA-ready environments.

MQ-CeNN does **not** implement physical quantum computation. It does not require a quantum processor, quantum circuit execution or state-vector simulation.

---

## Installation

### Standard installation

```bash
pip install -e .

This installs MQ-CeNN in editable development mode.

Development installation
pip install -e .[dev]

or:

pip install -r requirements-dev.txt
Backend behavior

MQ-CeNN is CPU-safe by default.

The default backend is:

NumPy / CPU

The framework includes a backend dispatcher that detects the execution environment.

from mq_cenn.backends import resolve_backend

print(resolve_backend("auto", "auto"))

On a CPU-only machine, the expected result is similar to:

BackendInfo(backend='numpy', device='cpu', cuda_available=False, cpp_available=False, cuda_backend_available=False)
CLI commands

After installation, the command mqcenn is available.

Environment check
mqcenn doctor
Install default CPU backend
mqcenn install-backend

Equivalent to:

mqcenn install-backend --cpu
Build C++ backend
mqcenn install-backend --cpp

This requires a C++17-compatible compiler.

Build CUDA backend
mqcenn install-backend --cuda

This requires:

NVIDIA GPU;
compatible NVIDIA driver;
CUDA Toolkit;
CUDA-compatible PyTorch build;
working compiler toolchain.

CUDA is never forced during standard installation.

Quickstart
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
    bridge_dim=8,
    cenn_hidden=8,
    cenn_epochs=1,
    batch_size=16,
    patience=1,
    stationarize=True,
    last_value_index=lookback - 1,
    backend="auto",
    device="auto",
    random_state=42,
)

model.fit(X[:-10], y[:-10])

pred = model.predict(X[-10:])

print(pred.shape)
print(model.trace_.backend, model.trace_.device)
Prediction intervals
pred, lower, upper = model.predict_interval(X[-10:])
Diagnostics
diagnostics = model.predict_with_diagnostics(X[-10:])

print(diagnostics.keys())
print(diagnostics["reliability"])
print(diagnostics["fallback_mask"])

Diagnostics include:

final prediction;
core prediction;
reliability score;
fallback mask;
conformal interval bounds;
teacher mean;
expert pool predictions.
Ablation suite
from mq_cenn import make_ablation_suite

suite = make_ablation_suite(
    n_features_per_expert=32,
    n_experts_per_kernel=1,
    cenn_epochs=1,
    backend="auto",
    device="auto",
)

print(suite.keys())

Available variants:

MQCeNN_full
MQCeNN_softmax_gate
MQCeNN_no_periodic_kernel
MQCeNN_gaussian_only
MQCeNN_strict_reliability
MQCeNN_no_fallback

The ablation suite does not automatically select the best model. Each estimator must be trained and evaluated under the same protocol.

Testing

Install development dependencies:

pip install -e .[dev]

Run the full test suite:

pytest

Run a specific test file:

pytest tests/test_validation.py

Current expected result:

49 passed
Scientific positioning

MQ-CeNN is positioned as a classical quantum-inspired forecasting framework.

The following statements are intentionally avoided:

quantum advantage;
physical quantum computation;
real quantum entanglement;
real quantum interference.

Instead, MQ-CeNN uses classical proxies:

Concept	Implementation
Quantum-inspired feature lifting	Multi-kernel random features
Cross-expert dependence	CrossExpertBridge
Interference-like behavior	SignedInterferenceGate
Reliability	ReliabilityCalibrator
Fail-safe behavior	Fallback strategy
Uncertainty	Split-conformal intervals
Public API
from mq_cenn import (
    MQCeNNRegressor,
    MQCeNNTrace,
    KernelSpec,
    DEFAULT_KERNEL_SPECS,
    make_ablation_suite,
)
License

This project is released under the MIT License.


---

Après avoir créé ces fichiers, fais :

```cmd
git status

Puis ajoute uniquement les bons fichiers :

git add README.md LICENSE .gitignore requirements.txt requirements-dev.txt
git add tests mq_cenn native pyproject.toml setup.py

Vérifie avant commit :

git status

Tu ne dois pas voir .venv/.

Puis :

git commit -m "Add project metadata requirements and documentation"
git push origin main
