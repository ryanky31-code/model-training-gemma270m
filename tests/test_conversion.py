import os
import sys
import tempfile
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.generate_synthetic_smoke import generate_synthetic_row


def test_smoke_row_and_csv_roundtrip():
    rows = [generate_synthetic_row(i) for i in range(5)]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tmp.csv")
        df.to_csv(p, index=False)
        df2 = pd.read_csv(p)
        assert len(df2) == 5
        assert "expected_throughput_mbps" in df2.columns
