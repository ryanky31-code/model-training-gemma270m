import csv
import subprocess
import tempfile
from pathlib import Path


def make_preds_csv(path: str):
    # simple CSV with numeric and categorical examples
    rows = [
        {"expected_throughput_mbps": "10", "pred": "12"},
        {"expected_throughput_mbps": "20", "pred": "18"},
        {"expected_throughput_mbps": "high", "pred": "high"},
        {"expected_throughput_mbps": "low", "pred": "medium"},
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["expected_throughput_mbps", "pred"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_evaluate_model_subprocess(tmp_path: Path):
    p = tmp_path / "preds.csv"
    make_preds_csv(str(p))
    proc = subprocess.run([
        "python",
        "scripts/evaluate_model.py",
        "--preds",
        str(p),
        "--target-field",
        "expected_throughput_mbps",
        "--pred-field",
        "pred",
    ], capture_output=True, text=True)
    assert proc.returncode == 0
    out = proc.stdout
    assert "Numeric targets" in out
    assert "Categorical targets" in out
