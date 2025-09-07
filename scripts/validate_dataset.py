#!/usr/bin/env python3
"""Basic dataset validation utilities."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def validate(path: Path):
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    # basic null checks
    nulls = df.isnull().sum()
    print("Nulls per column:")
    print(nulls[nulls > 0])
    # range checks
    if "expected_throughput_mbps" in df.columns:
        vals = df["expected_throughput_mbps"].dropna()
        print("throughput min/max:", vals.min(), vals.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to CSV file")
    args = parser.parse_args()
    validate(Path(args.path))


if __name__ == '__main__':
    main()
