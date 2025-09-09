"""Simple hyperparameter sweep helper (dry-run)

Runs `scripts/finetune_gemma_from_csv.py` in `--dry-run` mode over a small grid of
hyperparameters and saves the per-trial stdout/stderr and returncode to a JSON file.

This is intentionally lightweight and meant to be run locally or in CI with small max_rows.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run_trial(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--target-field", required=True, help="Target column name")
    parser.add_argument("--max-rows", type=int, default=50, help="Max rows to use for dry-run")
    parser.add_argument("--out", default="sweep_results.json", help="Output JSON file")
    parser.add_argument("--mode", choices=["full", "lora", "qlora"], default="lora")
    parser.add_argument("--lrs", nargs="*", type=float, default=[1e-4, 5e-5], help="Learning rates to try")
    parser.add_argument("--batches", nargs="*", type=int, default=[8, 16], help="Per-device batch sizes")
    parser.add_argument("--epochs", nargs="*", type=int, default=[1, 2], help="Epoch counts to try")
    args = parser.parse_args(argv)

    grid = list(itertools.product(args.lrs, args.batches, args.epochs))
    results: List[Dict[str, Any]] = []

    for lr, batch, epochs in grid:
        cmd = [
            "python",
            "scripts/finetune_gemma_from_csv.py",
            "--csv",
            str(args.csv),
            "--dry-run",
            "--max-rows",
            str(args.max_rows),
            "--target-field",
            args.target_field,
            "--mode",
            args.mode,
            "--learning-rate",
            str(lr),
            "--per-device-batch-size",
            str(batch),
            "--num-epochs",
            str(epochs),
        ]
        print(f"Running trial: lr={lr} batch={batch} epochs={epochs}")
        res = run_trial(cmd)
        res["params"] = {"learning_rate": lr, "batch_size": batch, "epochs": epochs, "mode": args.mode}
        results.append(res)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} trial results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
