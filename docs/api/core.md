# Core API

The `mq_cenn.core` package contains the internal building blocks used by the public estimator.

```text
mq_cenn/core/
├── kernels.py
├── experts.py
├── reliability.py
├── bridge.py
└── gate.py
```

## Kernel lifting

Main objects:

```python
from mq_cenn.core import KernelSpec, DEFAULT_KERNEL_SPECS, SpectralFeatureProjector
```

### `KernelSpec`

Defines a kernel-feature configuration.

Typical information:

- kernel name;
- feature dimension;
- scale;
- degree or periodicity parameters depending on the kernel type.

### `SpectralFeatureProjector`

Projects input vectors into random feature spaces.

This is the core of the quantum-inspired lifting idea: the model maps the original time-series window into several nonlinear feature spaces before fitting expert predictors.

## Experts

Main objects:

```python
from mq_cenn.core import KernelRidgeExpert, MultiKernelExpertPool
```

### `KernelRidgeExpert`

A ridge-style expert trained on one lifted feature space.

### `MultiKernelExpertPool`

A collection of experts built from multiple kernel families. Each expert produces an intermediate forecast, and the gate later combines them.

## Reliability

Main objects:

```python
from mq_cenn.core import NoveltyDetector, ReliabilityCalibrator
```

### `NoveltyDetector`

Estimates how far a test sample is from the training/calibration distribution.

### `ReliabilityCalibrator`

Transforms novelty or calibration information into a reliability score.

The reliability score is used to decide whether the core prediction should be trusted or replaced by a fallback prediction.

## Bridge

Main object:

```python
from mq_cenn.core import CrossExpertBridge
```

The bridge creates a representation that connects expert outputs and internal model states before gated aggregation.

## Gate

Main object:

```python
from mq_cenn.core import SignedInterferenceGate
```

The signed gate aggregates expert predictions using positive and negative interactions.

The goal is to model constructive and destructive expert interactions in a classical, differentiable way.

## Design principle

The core API is modular: each component should remain testable independently. This is essential for ablation studies and scientific reproducibility.
