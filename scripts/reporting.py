"""Simple reporting helpers

- Plot learning curves (MAE/RMSE vs dataset size) from `learning_curve_results.json`.
- If confusion matrices are present, render a heatmap (requires matplotlib).

This script is optional and will not fail if matplotlib is not installed; it prints a short summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_learning_curve(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text())


def summarize_learning_curve(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    # entries: list of trial result dicts with structure produced by run_learning_curve.py
    summary = {"runs": len(entries)}
    sizes = []
    maes = []
    rmses = []
    for e in entries:
        try:
            # try to extract from nested JSON stdout (dry-run stored stdout in earlier runs)
            out = e.get("stdout", "")
            # look for markers 'MAE=' or 'MAE' in json output
            # For now, we parse if e contains a 'metrics' field
            metrics = e.get("metrics") or {}
            if metrics:
                if "mae" in metrics:
                    maes.append(metrics["mae"])
                if "rmse" in metrics:
                    rmses.append(metrics["rmse"])
            # size/key
            max_rows = e.get("max_rows") or e.get("params", {}).get("max_rows") or e.get("params", {}).get("max-rows")
            if max_rows:
                sizes.append(int(max_rows))
        except Exception:
            continue
    summary["sizes"] = sizes
    summary["maes"] = maes
    summary["rmses"] = rmses
    return summary


def try_plot(entries: List[Dict[str, Any]], out_prefix: str = "reports/learning_curve") -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib or numpy not available — skipping plots")
        return

    pfx = Path(out_prefix)
    pfx.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_learning_curve(entries)
    sizes = summary.get("sizes", [])
    maes = summary.get("maes", [])
    rmses = summary.get("rmses", [])

    if sizes and maes:
        idx = np.argsort(sizes)
        sizes_arr = np.array(sizes)[idx]
        maes_arr = np.array(maes)[idx]
        plt.figure()
        plt.plot(sizes_arr, maes_arr, marker="o")
        plt.xlabel("dataset size")
        plt.ylabel("MAE")
        plt.title("Learning curve (MAE)")
        plt.grid(True)
        out_file = str(pfx) + "_mae.png"
        plt.savefig(out_file)
        print(f"Wrote {out_file}")
    if sizes and rmses:
        idx = np.argsort(sizes)
        sizes_arr = np.array(sizes)[idx]
        rmses_arr = np.array(rmses)[idx]
        plt.figure()
        plt.plot(sizes_arr, rmses_arr, marker="o", color="orange")
        plt.xlabel("dataset size")
        plt.ylabel("RMSE")
        plt.title("Learning curve (RMSE)")
        plt.grid(True)
        out_file = str(pfx) + "_rmse.png"
        plt.savefig(out_file)
        print(f"Wrote {out_file}")


def try_plot_confusion_from_file(path: str, out_prefix: str = "reports/confusion") -> bool:
    """Load an evaluation JSON file containing a 'categorical'->'confusion' dict and plot a heatmap.

    Returns True when plotting succeeded, False when plotting skipped (missing deps) or file missing.
    """
    p = Path(path)
    if not p.exists():
        print(f"Evaluation file not found: {path}")
        return False
    try:
        data = json.loads(p.read_text())
    except Exception:
        print(f"Failed to read JSON from {path}")
        return False

    cat = data.get("categorical") or {}
    confusion = cat.get("confusion")
    if not confusion:
        print("No confusion data found in evaluation JSON")
        return False

    # Attempt to import plotting libs
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib or numpy not available — skipping confusion plot")
        return False

    labels = sorted(confusion.keys())
    matrix = np.array([[confusion[g].get(p, 0) for p in labels] for g in labels], dtype=float)

    outp = Path(out_prefix)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("pred")
    plt.ylabel("gold")
    plt.title("Confusion matrix")
    out_file = str(outp) + "_confusion.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Wrote {out_file}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="learning_curve_results.json")
    parser.add_argument("--out-prefix", default="reports/learning_curve")
    args = parser.parse_args()

    try:
        entries = load_learning_curve(args.input)
    except FileNotFoundError:
        print(f"No learning curve file found at {args.input}")
        raise SystemExit(1)

    print(summarize_learning_curve(entries))
    try_plot(entries, out_prefix=args.out_prefix)
