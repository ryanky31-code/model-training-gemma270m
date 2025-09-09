import numpy as np
import pandas as pd

from scripts.baselines import run_baselines


def test_run_baselines_numeric():
    # simple numeric dataset
    df = pd.DataFrame({"expected_throughput_mbps": [10, 20, 30, 40, 50]})
    results = run_baselines(df, target_field="expected_throughput_mbps", test_size=0.4, seed=1)
    assert "mean" in results and "mae" in results["mean"]


def test_run_baselines_small_dataset():
    df = pd.DataFrame({"expected_throughput_mbps": [100]})
    results = run_baselines(df, target_field="expected_throughput_mbps", test_size=0.5, seed=2)
    # small test set may be empty — should return empty dict or results
    assert isinstance(results, dict)


def test_training_cli_dry_run_subprocess():
    import subprocess
    proc = subprocess.run([
        "python",
        "scripts/finetune_gemma_from_csv.py",
        "--csv",
        "synthetic_wifi_5ghz_outdoor_smoke.csv",
        "--dry-run",
        "--max-rows",
        "5",
        "--target-field",
        "expected_throughput_mbps",
    ], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Dry-run: prepared dataset" in proc.stdout
