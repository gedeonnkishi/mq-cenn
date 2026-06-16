# Quantum-Inspired Positioning

MQ-CeNN is a **quantum-inspired classical framework**. It borrows conceptual motifs from quantum machine learning, but it runs on classical hardware by default.

## What MQ-CeNN is

MQ-CeNN is:

- a classical machine-learning model;
- a random-feature and multi-kernel forecasting framework;
- a reliability-aware estimator;
- an engineering-oriented framework for time-series regression.

## What MQ-CeNN is not

MQ-CeNN is not:

- a quantum computer implementation;
- a quantum circuit simulator by default;
- proof of quantum advantage;
- a replacement for rigorous baselines.

## Why use the term quantum-inspired?

The term is justified only at the level of modeling inspiration:

- high-dimensional feature lifting;
- superposition-like aggregation of multiple feature views;
- constructive and destructive interference-inspired gating;
- probabilistic reliability interpretation.

These are classical analogies. They do not imply quantum speedup.

## Correct scientific claim

A cautious claim is:

> MQ-CeNN is a quantum-inspired random-feature framework for robust time-series forecasting under non-stationarity.

A claim to avoid:

> MQ-CeNN achieves quantum advantage.

That statement would require evidence from actual quantum hardware or a complexity-theoretic argument, neither of which is part of the current framework.

## Recommended evaluation

MQ-CeNN should be evaluated under a strict benchmark protocol:

- chronological splits;
- train-only normalization;
- multiple horizons;
- multiple lookbacks;
- multiple seeds;
- strong baselines;
- ablation studies;
- transparent negative results.

## Publication posture

The best scientific posture is benchmark-first:

1. define the method;
2. define the reproducible protocol;
3. compare fairly;
4. analyze when the method works;
5. analyze when it fails;
6. avoid overclaiming.
