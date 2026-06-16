# Signed Interference Gate

The signed interference gate aggregates predictions from multiple experts using positive and negative interactions.

## Expert prediction pool

Assume the expert pool returns:

```text
p(x) = [p_1(x), p_2(x), ..., p_K(x)]
```

A standard gate might learn positive normalized weights:

```text
y_hat = sum_k w_k p_k(x)
```

with:

```text
w_k >= 0
sum_k w_k = 1
```

## Signed aggregation

MQ-CeNN allows signed interactions:

```text
y_hat = sum_k a_k(x) p_k(x)
```

where some coefficients may act constructively and others destructively.

This is inspired by interference-like behavior, but it is implemented as a classical differentiable aggregation mechanism.

## Why signed interactions?

In a heterogeneous expert pool:

- some experts may be useful only in certain regimes;
- some experts may systematically overestimate;
- some may be useful as corrective signals;
- negative contribution can reduce biased predictions.

## Scientific interpretation

The signed gate should not be described as a quantum circuit. It is better understood as:

- adaptive expert aggregation;
- signed mixture-of-experts;
- interference-inspired feature combination.

## Ablation

The signed gate must be justified experimentally.

The key comparison is:

```text
MQCeNN_full
vs
MQCeNN_softmax_gate
```

If the softmax gate performs better, the signed gate is not helping on that dataset. That result should be reported honestly.

## Failure modes

The signed gate can fail when:

- the dataset is too small;
- the signal is too simple;
- experts are highly correlated;
- calibration is weak;
- the gate overfits.

This is why the framework includes reliability calibration and fallback behavior.
