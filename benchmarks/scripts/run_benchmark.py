from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mq_cenn import MQCeNNRegressor  # noqa: E402


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "default_regression",
        "seed": 42,
        "output_dir": "benchmarks/results",
    },
    "data": {
        "kind": "synthetic_nonstationary",
        "n_samples": 220,
        "lookback": 24,
        "noise": 0.05,
        "train_fraction": 0.70,
        "calibration_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "model": {
        "n_features_per_expert": 32,
        "n_experts_per_kernel": 1,
        "bridge_dim": 8,
        "cenn_hidden": 8,
        "cenn_epochs": 1,
        "batch_size": 16,
        "patience": 1,
        "stationarize": True,
        "backend": "auto",
        "device": "auto",
    },
    "baselines": {
        "persistence": True,
        "moving_average": True,
        "ridge": True,
        "mq_cenn": True,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Falling back to built-in defaults.")
        return DEFAULT_CONFIG

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    return deep_update(DEFAULT_CONFIG, loaded)


def make_synthetic_nonstationary(
    n_samples: int,
    lookback: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)

    trend = 0.002 * t
    seasonal = np.sin(t / 6.0) + 0.35 * np.sin(t / 17.0)
    regime = np.where(t >= n_samples * 0.55, 0.35, 0.0)
    series = seasonal + trend + regime + noise * rng.normal(size=n_samples)

    X = []
    y = []

    for i in range(lookback, n_samples):
        X.append(series[i - lookback:i])
        y.append(series[i])

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    train_cal_fraction = float(train_fraction) + float(calibration_fraction)
    train_cal_end = int(np.floor(n * train_cal_fraction))
    train_cal_end = min(max(train_cal_end, 20), n - 5)

    return X[:train_cal_end], y[:train_cal_end], X[train_cal_end:], y[train_cal_end:]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def mase(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = float(np.mean(np.abs(np.diff(y_train))))
    if scale <= 1e-12:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def evaluate_model(
    name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    backend: str = "none",
    device: str = "none",
) -> Dict[str, Any]:
    return {
        "model": name,
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mase": mase(y_train, y_test, y_pred),
        "r2": r2_score_np(y_test, y_pred),
        "backend": backend,
        "device": device,
    }


def write_results(results: List[Dict[str, Any]], output_dir: Path, experiment_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{experiment_name}_results.csv"
    json_path = output_dir / f"{experiment_name}_results.json"

    fieldnames = ["model", "mae", "rmse", "mase", "r2", "backend", "device"]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


def run(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    seed = int(config["experiment"]["seed"])
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    baseline_cfg = config["baselines"]

    lookback = int(data_cfg["lookback"])
    X, y = make_synthetic_nonstationary(
        n_samples=int(data_cfg["n_samples"]),
        lookback=lookback,
        noise=float(data_cfg["noise"]),
        seed=seed,
    )

    X_train, y_train, X_test, y_test = chronological_split(
        X,
        y,
        train_fraction=float(data_cfg["train_fraction"]),
        calibration_fraction=float(data_cfg["calibration_fraction"]),
    )

    results: List[Dict[str, Any]] = []

    if baseline_cfg.get("persistence", True):
        pred = X_test[:, lookback - 1]
        results.append(evaluate_model("Persistence", y_train, y_test, pred))

    if baseline_cfg.get("moving_average", True):
        pred = X_test.mean(axis=1)
        results.append(evaluate_model("MovingAverage", y_train, y_test, pred))

    if baseline_cfg.get("ridge", True):
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)
        results.append(evaluate_model("Ridge", y_train, y_test, pred))

    if baseline_cfg.get("mq_cenn", True):
        model_cfg["last_value_index"] = lookback - 1
        model_cfg["random_state"] = seed

        model = MQCeNNRegressor(**model_cfg)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append(
            evaluate_model(
                "MQCeNN",
                y_train,
                y_test,
                pred,
                backend=model.trace_.backend,
                device=model.trace_.device,
            )
        )

    results = sorted(results, key=lambda row: row["mae"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MQ-CeNN regression benchmark.")
    parser.add_argument(
        "--config",
        default="benchmarks/configs/default_regression.yaml",
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run(config)

    print("Benchmark results")
    print("-----------------")
    for row in results:
        print(
            f"{row['model']:16s}",
            f"MAE={row['mae']:.6f}",
            f"RMSE={row['rmse']:.6f}",
            f"MASE={row['mase']:.6f}",
            f"R2={row['r2']:.6f}",
            f"backend={row['backend']}",
            f"device={row['device']}",
        )

    output_dir = ROOT / config["experiment"]["output_dir"]
    write_results(results, output_dir, config["experiment"]["name"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
