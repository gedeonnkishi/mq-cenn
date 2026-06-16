# Ablation API

Ablation studies evaluate which components of MQ-CeNN actually contribute to performance.

The public helper is:

```python
from mq_cenn import make_ablation_suite
```

## Basic usage

```python
suite = make_ablation_suite(
    n_features_per_expert=32,
    n_experts_per_kernel=1,
    bridge_dim=8,
    cenn_hidden=8,
    cenn_epochs=1,
    batch_size=16,
    patience=1,
    stationarize=True,
    last_value_index=23,
    backend="auto",
    device="auto",
    random_state=42,
)

for name, model in suite.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, pred[:5])
```

## Default variants

Typical variants include:

```text
MQCeNN_full
MQCeNN_softmax_gate
MQCeNN_no_periodic_kernel
MQCeNN_gaussian_only
MQCeNN_strict_reliability
MQCeNN_no_fallback
```

## Interpretation

### `MQCeNN_full`

Complete model with all core mechanisms enabled.

### `MQCeNN_softmax_gate`

Uses a simpler softmax-style gate. This tests whether the signed interference gate is useful.

### `MQCeNN_no_periodic_kernel`

Removes the periodic kernel family. This checks whether periodic feature lifting contributes to the task.

### `MQCeNN_gaussian_only`

Keeps only Gaussian-style features. This tests whether multi-kernel diversity is useful.

### `MQCeNN_strict_reliability`

Uses stricter reliability behavior. This tests the sensitivity of fallback decisions.

### `MQCeNN_no_fallback`

Disables fallback. This tests whether fallback improves robustness.

## Scientific reporting

Ablation results should be reported with:

- same dataset split;
- same seeds;
- same metrics;
- mean and standard deviation across seeds;
- training time when relevant.

A component should not be claimed useful unless the ablation shows consistent improvement.
