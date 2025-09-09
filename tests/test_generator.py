import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.generate_synthetic_large import apply_stratified_resample


def make_rows(groups):
    rows = []
    for label, count in groups.items():
        for i in range(count):
            rows.append({"label": label, "value": f"{label}-{i}"})
    return rows


def test_deterministic_with_seed():
    rows = make_rows({"A": 2, "B": 3})
    out1 = apply_stratified_resample(rows, stratify_by="label", balance_mode="oversample", seed=42)
    out2 = apply_stratified_resample(rows, stratify_by="label", balance_mode="oversample", seed=42)
    assert out1 == out2


def test_oversample_targets_max_group_size():
    rows = make_rows({"A": 1, "B": 4})
    out = apply_stratified_resample(rows, stratify_by="label", balance_mode="oversample", seed=1)
    counts = {}
    for r in out:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert counts["A"] == 4
    assert counts["B"] == 4


def test_undersample_targets_min_group_size():
    rows = make_rows({"A": 2, "B": 5})
    out = apply_stratified_resample(rows, stratify_by="label", balance_mode="undersample", seed=7)
    counts = {}
    for r in out:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert counts["A"] == 2
    assert counts["B"] == 2
