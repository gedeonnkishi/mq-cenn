# Reliability and Fallback

MQ-CeNN includes reliability calibration to decide when the model should trust its core prediction.

## Motivation

In time-series forecasting, models often fail during:

- distribution shift;
- regime changes;
- outliers;
- rare events;
- extrapolation beyond the training distribution.

A reliability-aware model should know when its prediction may be unsafe.

## Reliability score

The reliability module estimates a score:

```text
r(x) in [0, 1]
```

High reliability means the input looks close to the training/calibration distribution.

Low reliability means the model may be extrapolating or facing an unusual state.

## Fallback strategy

If reliability is below a threshold, MQ-CeNN can replace the core prediction with a safer fallback.

Typical fallback strategies include:

```text
stable_ridge
teacher_mean
persistence
```

## Fallback mask

During diagnostics, the model exposes:

```python
diagnostics["fallback_mask"]
```

This shows which predictions used fallback.

## Why fallback can help

Fallback is useful when the full model is expressive but unstable under uncertainty. A simple predictor can be more robust in low-confidence regions.

For example, if the model has low reliability on a test window, using persistence or a stable ridge estimate may reduce large errors.

## Ablation

The key ablation is:

```text
MQCeNN_full
vs
MQCeNN_no_fallback
```

If `MQCeNN_no_fallback` performs worse, fallback is contributing to robustness.

If it performs better, the fallback threshold or calibration may need adjustment.

## Conformal interval

MQ-CeNN can also produce prediction intervals using an absolute residual radius estimated on calibration data.

The interval is:

```text
[y_hat - q, y_hat + q]
```

where `q` is the calibrated residual radius.

## Scientific caution

Reliability and fallback do not guarantee correctness. They are practical mechanisms for robust forecasting and must be validated empirically.

Recommended reporting:

- fallback rate;
- mean reliability;
- calibration interval radius;
- error with fallback;
- error without fallback;
- behavior under distribution shift.
