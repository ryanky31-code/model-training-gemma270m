import argparse
import csv
import json
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test(values: Iterable, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    arr = np.array(list(values))
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([])
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(math.ceil(n * (1 - test_size)))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return arr[train_idx], arr[test_idx]


def random_baseline_predict(train_values: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(train_values) == 0:
        return np.array([])
    return rng.choice(train_values, size=n)


def frequency_baseline_predict(train_values: np.ndarray, n: int) -> np.ndarray:
    if len(train_values) == 0:
        return np.array([])
    most_common = Counter(train_values).most_common(1)[0][0]
    return np.array([most_common] * n)


def mean_baseline_predict(train_values: np.ndarray, n: int) -> np.ndarray:
    if len(train_values) == 0:
        return np.array([])
    # If numeric, return mean; otherwise fallback to mode
    try:
        vals = train_values.astype(float)
        mean = float(np.mean(vals))
        return np.array([mean] * n)
    except Exception:
        return frequency_baseline_predict(train_values, n)


def evaluate_numeric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_categorical(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    return {"accuracy": acc}


def is_numeric_array(arr: np.ndarray) -> bool:
    try:
        _ = arr.astype(float)
        return True
    except Exception:
        return False


def run_baselines(df: pd.DataFrame, target_field: str = "expected_throughput_mbps", test_size: float = 0.2, seed: int = 0) -> Dict[str, Dict[str, float]]:
    values = df[target_field].dropna().values
    train, test = split_train_test(values, test_size=test_size, seed=seed)
    if len(test) == 0:
        return {}

    results: Dict[str, Dict[str, float]] = {}

    # Random baseline (sample from empirical distribution)
    rand_pred = random_baseline_predict(train, len(test), seed=seed)
    # Frequency baseline (most common)
    freq_pred = frequency_baseline_predict(train, len(test))
    # Mean baseline (numeric)
    mean_pred = mean_baseline_predict(train, len(test))

    if is_numeric_array(test):
        results["random"] = evaluate_numeric(test, rand_pred)
        results["frequency"] = evaluate_numeric(test, freq_pred)
        results["mean"] = evaluate_numeric(test, mean_pred)
    else:
        results["random"] = evaluate_categorical(test, rand_pred)
        results["frequency"] = evaluate_categorical(test, freq_pred)
        results["mean"] = evaluate_categorical(test, mean_pred)

    return results


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with target field")
    parser.add_argument("--target-field", default="expected_throughput_mbps")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output file (CSV or JSON). If omitted prints to stdout")
    args = parser.parse_args()

    df = load_csv(args.csv)
    results = run_baselines(df, target_field=args.target_field, test_size=args.test_size, seed=args.seed)

    if args.out:
        if args.out.endswith(".csv"):
            # flatten results
            rows = []
            for k, metrics in results.items():
                row = {"baseline": k}
                row.update(metrics)
                rows.append(row)
            keys = ["baseline"] + sorted(next(iter(results.values())).keys())
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    cli()
