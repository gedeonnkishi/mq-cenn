# Kernel Lifting

Kernel lifting maps input vectors into nonlinear feature spaces before prediction.

For a time-series window:

```text
x = [x_{t-W}, ..., x_{t-1}]
```

MQ-CeNN creates several lifted representations:

```text
phi_1(x), phi_2(x), ..., phi_K(x)
```

Each representation corresponds to a different kernel family or random-feature transformation.

## Motivation

A single linear model may fail to capture nonlinear temporal structure. Kernel lifting allows simple experts to operate in richer feature spaces.

This is especially useful when the data contains:

- periodic behavior;
- local smoothness;
- nonlinear interactions;
- regime changes;
- noise and outliers.

## Expert-level prediction

Each lifted representation feeds an expert:

```text
f_k(x) = expert_k(phi_k(x))
```

The model therefore obtains a pool of predictions:

```text
[f_1(x), f_2(x), ..., f_K(x)]
```

These predictions are later aggregated by the gate.

## Multi-kernel design

Using multiple kernels is useful because different time-series structures require different inductive biases.

Examples:

- Gaussian-like features for smooth nonlinear structure;
- periodic features for cyclic patterns;
- polynomial features for interactions and trends.

## Scientific risk

More kernels do not automatically mean better performance.

A richer feature space can:

- improve expressiveness;
- increase variance;
- overfit small datasets;
- perform worse than Ridge on simple tasks.

That is why MQ-CeNN includes ablation variants such as:

```text
MQCeNN_gaussian_only
MQCeNN_no_periodic_kernel
MQCeNN_full
```

## Reporting recommendation

Kernel-lifting results should be reported with:

- number of random features;
- kernel families used;
- number of experts;
- seed protocol;
- runtime cost;
- ablation comparison.
