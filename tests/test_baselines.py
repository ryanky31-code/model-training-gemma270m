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
