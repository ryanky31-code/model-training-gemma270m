import os
import tempfile
import pandas as pd
import subprocess
import sys

# Ensure tests run from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.generate_synthetic_smoke import generate_synthetic_row


def test_finetune_dryrun_creates_dataset():
    # create a tiny CSV
    rows = [generate_synthetic_row(i) for i in range(3)]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'tmp.csv')
        df.to_csv(p, index=False)
        # Run the finetune script in dry-run mode
        cmd = [sys.executable, 'scripts/finetune_gemma_from_csv.py', '--csv', p, '--dry-run', '--max-rows', '3']
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        assert res.returncode == 0
        assert 'Dry-run: prepared dataset' in res.stdout
