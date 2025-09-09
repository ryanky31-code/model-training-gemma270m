"""Small evaluation harness

Computes:
- Numeric targets: MAE, RMSE
- Categorical targets: accuracy, confusion matrix (counts)

Usage (CSV):
    python scripts/evaluate_model.py --preds preds.csv --target-field expected_throughput_mbps --pred-field pred

The CSV must contain at least the ground-truth target column and a prediction column.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def read_csv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            yield r


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def evaluate_numeric(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    if not pairs:
        return {"mae": math.nan, "rmse": math.nan, "n": 0}
    diffs = [abs(y - yhat) for y, yhat in pairs]
    mse = sum((y - yhat) ** 2 for y, yhat in pairs) / len(pairs)
    return {"mae": sum(diffs) / len(diffs), "rmse": math.sqrt(mse), "n": len(pairs)}


def evaluate_categorical(pairs: List[Tuple[str, str]]) -> Dict[str, object]:
    if not pairs:
        return {"accuracy": math.nan, "n": 0, "confusion": {}}
    n = len(pairs)
    correct = sum(1 for y, yhat in pairs if y == yhat)
    accuracy = correct / n
    labels = set([y for y, _ in pairs] + [yhat for _, yhat in pairs])
    label_list = sorted(labels)
    # confusion counts: gold -> pred -> count
    confusion: Dict[str, Dict[str, int]] = {l: {l2: 0 for l2 in label_list} for l in label_list}
    for y, yhat in pairs:
        confusion[y][yhat] += 1
    return {"accuracy": accuracy, "n": n, "confusion": confusion}


def load_pairs_from_csv(path: str, target_field: str, pred_field: str) -> Tuple[List[Tuple[float, float]], List[Tuple[str, str]]]:
    numeric: List[Tuple[float, float]] = []
    categorical: List[Tuple[str, str]] = []
    for row in read_csv_rows(path):
        if target_field not in row or pred_field not in row:
            continue
        t = row[target_field]
        p = row[pred_field]
        if is_number(t) and is_number(p):
            numeric.append((float(t), float(p)))
        else:
            categorical.append((t, p))
    return numeric, categorical


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="CSV/JSONL file with predictions and target")
    parser.add_argument("--target-field", required=True, help="Name of the ground-truth target column")
    parser.add_argument("--pred-field", required=True, help="Name of the prediction column")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--json-output", help="Write summary to JSON file")
    args = parser.parse_args(argv)

    numeric, categorical = load_pairs_from_csv(args.preds, args.target_field, args.pred_field)
    out = {}
    if numeric:
        out["numeric"] = evaluate_numeric(numeric)
    if categorical:
        out["categorical"] = evaluate_categorical(categorical)

    # Print a human-friendly summary
    if "numeric" in out:
        ninfo = out["numeric"]
        print(f"Numeric targets (n={ninfo['n']}): MAE={ninfo['mae']:.4f}, RMSE={ninfo['rmse']:.4f}")
    if "categorical" in out:
        cinfo = out["categorical"]
        print(f"Categorical targets (n={cinfo['n']}): accuracy={cinfo['accuracy']:.4f}")
        print("Confusion matrix (gold rows -> pred cols):")
        labels = sorted(cinfo["confusion"].keys())
        header = ["gold\\pred"] + labels
        print("\t".join(header))
        for g in labels:
            row = [g] + [str(cinfo["confusion"][g][p]) for p in labels]
            print("\t".join(row))

    if args.json_output:
        with open(args.json_output, "w") as fh:
            json.dump(out, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
