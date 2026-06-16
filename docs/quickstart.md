# Quickstart

This guide shows a minimal MQ-CeNN regression workflow.

## 1. Generate a synthetic time series

```python
import numpy as np

def make_dataset(n=120, lookback=24, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    series = np.sin(t / 6.0) + 0.05 * rng.normal(size=n)

    X = []
    y = []

    for i in range(lookback, n):
        X.append(series[i - lookback:i])
        y.append(series[i])

    return np.asarray(X), np.asarray(y)
```

## 2. Fit MQ-CeNN

```python
from mq_cenn import MQCeNNRegressor

lookback = 24
X, y = make_dataset(lookback=lookback)

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

print(pred)
print(model.trace_)
```

## 3. Prediction intervals

```python
pred, lower, upper = model.predict_interval(X[-10:])
```

MQ-CeNN uses a conformal-style absolute residual radius estimated from calibration data.

## 4. Diagnostics

```python
diagnostics = model.predict_with_diagnostics(X[-10:])

print(diagnostics.keys())
print(diagnostics["prediction"])
print(diagnostics["reliability"])
print(diagnostics["fallback_mask"])
```

Diagnostics expose information about the core prediction, reliability score, fallback decisions, prediction intervals, teacher mean, and expert pool predictions.

## 5. Recommended experimental protocol

For serious benchmarking:

1. Use chronological train/validation/test splits.
2. Normalize using training data only.
3. Evaluate multiple seeds.
4. Evaluate multiple lookbacks and horizons.
5. Compare against strong baselines.
6. Report negative results honestly.
