# MQ-CeNN: Multi-Quantum Cellular Neural Network

Official Python implementation of the **MQ-CeNN** (Multi-Quantum Cellular Neural Network) framework, an open-source library designed for advanced, non-stationary time series forecasting.

MQ-CeNN combines quantum-informed functional representation spaces (via Quantum Random Fourier Features) with the localized spatial-temporal dynamical processing of Cellular Neural Networks (CeNN). This hybrid architecture allows the model to capture complex, multi-scale temporal dependencies without data leakage.

## 🚀 Key Features

* **Quantum-Informed Representation:** Emulated quantum Hilbert space projections via Random Fourier Features (RFF) to build a robust pool of independent quantum experts.
* **Cellular Neural Gating Network:** Uses localized 1D convolutions and non-linear states to compute dynamic attention weights across the expert pool.
* **Rigorous Mathematical Isolation:** Strict chronological data splitting (70/10/20) and automated rolling-window stationarization protocols to eliminate any risk of historical data snooping.
* **Research & Benchmark Ready:** Fully compatible with standard scikit-learn API (`fit`, `predict`), optimized for execution with PyTorch/CUDA accelerators.

## 📦 Installation

You can install the library directly into your Python environment or Kaggle/Colab notebooks using `pip`:

```bash
pip install git+[https://github.com/gedeonnkishi/mq-cenn.git](https://github.com/gedeonnkishi/mq-cenn.git)
