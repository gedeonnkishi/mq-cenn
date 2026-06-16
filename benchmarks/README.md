# MQ-CeNN Benchmarks

This folder contains reproducible benchmark scripts for MQ-CeNN.

The benchmark layer has three goals:

1. compare MQ-CeNN with simple and classical baselines;
2. run ablation variants under the same protocol;
3. export results as CSV/JSON files that can be summarized later.

The default scripts use a synthetic non-stationary time series so that the benchmark can run on any machine without external data.

---

## Structure

```text
benchmarks/
├── README.md
├── configs/
│   ├── default_regression.yaml
│   └── ablation.yaml
├── scripts/
│   ├── run_benchmark.py
│   ├── run_ablation.py
│   └── summarize_results.py
└── results/
    └── .gitkeep
```

---

## Run the default benchmark

From the repository root:

```bash
python benchmarks/scripts/run_benchmark.py
```

This creates:

```text
benchmarks/results/default_regression_results.csv
benchmarks/results/default_regression_results.json
```

---

## Run the ablation benchmark

```bash
python benchmarks/scripts/run_ablation.py
```

This creates:

```text
benchmarks/results/ablation_results.csv
benchmarks/results/ablation_results.json
```

---

## Summarize all results

```bash
python benchmarks/scripts/summarize_results.py
```

This scans `benchmarks/results/` and prints a compact ranking by MAE and RMSE.

---

## Optional configuration

You can pass an explicit config file:

```bash
python benchmarks/scripts/run_benchmark.py --config benchmarks/configs/default_regression.yaml
python benchmarks/scripts/run_ablation.py --config benchmarks/configs/ablation.yaml
```

If PyYAML is installed, YAML configuration files are parsed automatically.
If PyYAML is not installed, the scripts fall back to safe built-in defaults.

---

## Scientific discipline

Benchmark results should be interpreted cautiously:

- use fixed seeds;
- keep train/calibration/test splits chronological;
- report all baselines, not only the best MQ-CeNN result;
- do not claim quantum advantage;
- report failures, runtime and fallback behavior when relevant.
