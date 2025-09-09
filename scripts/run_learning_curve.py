#!/usr/bin/env python3
"""Run small learning-curve style dry-run experiments and record results.

This script uses the existing training CLI in dry-run mode to validate the pipeline
for increasing dataset sizes and records simple metrics (train/test counts) in JSON.
"""
import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def run_dry_run(csv_path: str, max_rows: int, target_field: str, seed: int = 0) -> dict:
    cmd = [
        "python",
        "scripts/finetune_gemma_from_csv.py",
        "--csv",
        csv_path,
        "--dry-run",
        "--max-rows",
        str(max_rows),
        "--target-field",
        target_field,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"max_rows": max_rows, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="synthetic_wifi_5ghz_outdoor_smoke.csv")
    parser.add_argument("--target-field", default="expected_throughput_mbps")
    parser.add_argument("--out", default="learning_curve_results.json")
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 200, 1000])
    args = parser.parse_args()

    results: List[dict] = []
    for s in args.sizes:
        r = run_dry_run(args.csv, s, args.target_field)
        results.append(r)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
