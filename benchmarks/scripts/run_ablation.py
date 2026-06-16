from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mq_cenn import make_ablation_suite  # noqa: E402
from run_benchmark import (  # noqa: E402
    DEFAULT_CONFIG,
    chronological_split,
    deep_update,
    evaluate_model,
    load_config,
    make_synthetic_nonstationary,
)


ABLATION_DEFAULT_CONFIG: Dict[str, Any] = deep_update(
    DEFAULT_CONFIG,
    {
        "experiment": {
            "name": "ablation",
            "seed": 42,
            "output_dir": "benchmarks/results",
        }
    },
)


def load_ablation_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return ABLATION_DEFAULT_CONFIG

    config = load_config(path)
    return deep_update(ABLATION_DEFAULT_CONFIG, config)


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

    model_cfg["last_value_index"] = lookback - 1
    model_cfg["random_state"] = seed

    suite = make_ablation_suite(**model_cfg)

    results: List[Dict[str, Any]] = []

    for name, model in suite.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append(
            evaluate_model(
                name,
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
    parser = argparse.ArgumentParser(description="Run MQ-CeNN ablation benchmark.")
    parser.add_argument(
        "--config",
        default="benchmarks/configs/ablation.yaml",
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    config = load_ablation_config(args.config)
    results = run(config)

    print("Ablation results")
    print("----------------")
    for row in results:
        print(
            f"{row['model']:28s}",
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
