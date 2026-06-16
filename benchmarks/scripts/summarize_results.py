from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("inf")


def summarize(results_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for path in sorted(results_dir.glob("*_results.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["source_file"] = path.name
            rows.append(row)

    return sorted(rows, key=lambda row: (as_float(row.get("mae", "inf")), as_float(row.get("rmse", "inf"))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MQ-CeNN benchmark results.")
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results",
        help="Directory containing *_results.csv files.",
    )
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return 1

    rows = summarize(results_dir)

    if not rows:
        print(f"No result files found in: {results_dir}")
        return 0

    print("Benchmark summary")
    print("-----------------")
    for rank, row in enumerate(rows, start=1):
        print(
            f"#{rank:02d}",
            f"{row.get('model', ''):28s}",
            f"MAE={as_float(row.get('mae', 'inf')):.6f}",
            f"RMSE={as_float(row.get('rmse', 'inf')):.6f}",
            f"MASE={as_float(row.get('mase', 'inf')):.6f}",
            f"R2={as_float(row.get('r2', 'inf')):.6f}",
            f"source={row.get('source_file', '')}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
