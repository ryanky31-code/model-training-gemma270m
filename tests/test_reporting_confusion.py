import json
from pathlib import Path

from scripts import reporting


def test_try_plot_confusion_smoke(tmp_path: Path):
    # create a small eval JSON with categorical/confusion structure
    data = {
        "categorical": {
            "accuracy": 0.5,
            "n": 4,
            "confusion": {
                "high": {"high": 1, "medium": 0},
                "low": {"low": 1, "medium": 1},
                "medium": {"medium": 1, "low": 0}
            }
        }
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data))
    # Function returns True if plotted, False if matplotlib missing or other issues
    res = reporting.try_plot_confusion_from_file(str(p), out_prefix=str(tmp_path / "out"))
    assert isinstance(res, bool)
