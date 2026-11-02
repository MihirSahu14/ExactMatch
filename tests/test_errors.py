"""Error-analysis check. Run: python -m tests.test_errors"""
import json
import tempfile
from pathlib import Path

from exactmatch import data, errors

assert errors.codes("Sony KDL-46V5100 46in TV") == {"KDL46V5100"}
assert errors.codes("WD 1TB USB 3.0") == set()

qs = data.load_queries("valid")
matched = next(q for q in qs if len(q.gold_ids) >= 2)
unmatched = next(q for q in qs if not q.gold_ids)
g1, g2 = sorted(matched.gold_ids)[:2]
other = next(p.id for p in data.load_catalog() if p.id not in matched.gold_ids)
rows = [  # one hit, one miss that was retrieved, one false match; plus a false match on a no-match query
    {"query_id": matched.record.id, "gold_ids": sorted(matched.gold_ids), "candidate_ids": [g1, g2, other],
     "returned_ids": [g1, other], "error": None},
    {"query_id": unmatched.record.id, "gold_ids": [], "candidate_ids": [other], "returned_ids": [other],
     "error": None},
]
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "20260101-000000_s0_valid"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"system": "s0", "split": "valid"}))
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    md = errors.analyze(d)
    assert "product not in catalog" in md and "missed:" in md and "false match:" in md, md
    (d / "config.json").write_text(json.dumps({"system": "s0", "split": "test"}))
    try:
        errors.analyze(d)
        raise AssertionError("test run must be refused")
    except SystemExit:
        pass
print("ok")
