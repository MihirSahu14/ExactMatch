"""Report check. Run: python -m tests.test_report"""
import json
import tempfile
from pathlib import Path

from exactmatch import report

with tempfile.TemporaryDirectory() as tmp:
    for name, system, f1 in [("20260101-000000_s0_test", "s0", 0.5), ("20260101-000001_s1_test", "s1", 0.1),
                             ("20260101-000002_s1_test", "s1", 0.7)]:  # later s1 run must win
        d = Path(tmp) / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"system": system, "split": "test", "limit": None}))
        (d / "metrics.json").write_text(json.dumps({
            "queries": 10, "queries_no_match": 3, "f1": f1, "precision": f1, "recall": f1, "no_match_acc": 1.0,
            "recall_at_k": 0.9, "k": 20, "false_matches": 1, "missed_matches": 2, "cost_per_1k_usd": 0.05,
            "latency_p95_s": 1.234, "errors": 0}))
    md = report.build(Path(tmp))
    assert "+0.200" in md and "0.100" not in md and "## test (full)" in md, md
    made = report.make_charts(Path(tmp), Path(tmp) / "charts")
    assert {"f1_by_system.png", "cost_vs_f1.png", "quality_breakdown.png"} <= set(made), made
    assert all((Path(tmp) / "charts" / f).stat().st_size > 0 for f in made)
print("ok")
