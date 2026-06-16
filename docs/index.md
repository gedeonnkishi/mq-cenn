# MQ-CeNN Documentation

**MQ-CeNN** is a quantum-inspired machine-learning framework for time-series regression and forecasting. It combines multi-kernel feature lifting, expert aggregation, a signed interference-inspired gate, reliability calibration, and fallback mechanisms.

The project is designed with a conservative scientific position:

- MQ-CeNN is **quantum-inspired**, not a claim of quantum advantage.
- The default implementation is CPU-safe and uses a NumPy backend.
- Native C++ and CUDA backends are optional acceleration paths.
- Results must be compared against strong classical baselines.

## Main components

```text
mq_cenn/
├── core/
│   ├── kernels.py
│   ├── experts.py
│   ├── reliability.py
│   ├── bridge.py
│   └── gate.py
├── estimators/
│   └── regressor.py
├── ablation/
│   └── suite.py
├── backends/
│   ├── dispatcher.py
│   ├── numpy_backend.py
│   ├── cpp_backend.py
│   └── cuda_backend.py
└── cli/
    └── main.py
```

## Documentation map

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- API:
  - [Regressor API](api/regressor.md)
  - [Core API](api/core.md)
  - [Ablation API](api/ablation.md)
- Theory:
  - [Quantum-inspired positioning](theory/qml_inspired_positioning.md)
  - [Kernel lifting](theory/kernel_lifting.md)
  - [Signed interference gate](theory/signed_interference_gate.md)
  - [Reliability and fallback](theory/reliability_fallback.md)

## Scientific positioning

MQ-CeNN should be evaluated as a classical, quantum-inspired random-feature framework. It must be benchmarked against simple and strong baselines such as:

- Persistence;
- Moving average;
- Ridge regression;
- HistGradientBoosting;
- ExtraTrees;
- MLP;
- recurrent or temporal neural models when appropriate.

A negative result is scientifically useful if the benchmark is rigorous and reproducible.
