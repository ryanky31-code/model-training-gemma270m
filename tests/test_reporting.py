import json
from pathlib import Path

from scripts import reporting


def test_summarize_learning_curve_optional():
    p = Path("learning_curve_results.json")
    if not p.exists():
        # nothing to test in CI — mark as skipped by returning
        return
    entries = json.loads(p.read_text())
    summary = reporting.summarize_learning_curve(entries)
    assert isinstance(summary, dict)
    assert "runs" in summary
