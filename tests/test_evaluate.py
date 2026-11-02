"""Scorer check. Run: python -m tests.test_evaluate"""
import json
import sys
import tempfile
import types
from pathlib import Path

from exactmatch import data, evaluate
from exactmatch.schema import Match, Result


def row(gold, got, cands, latency=1.0):
    return {"gold_ids": gold, "returned_ids": got, "candidate_ids": cands,
            "cost_usd": 0.001, "tokens": 10, "latency_s": latency, "error": None}


# 1 hit, 1 false match, 1 miss on query 1; query 2 correctly returns nothing
m = evaluate.score([row(["a", "b"], ["a", "c"], ["a", "b", "c"]), row([], [], ["x"], latency=3.0)], k=20)
assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5), m
assert m["exact_set_acc"] == 0.5 and m["no_match_acc"] == 1.0 and m["recall_at_k"] == 1.0, m
assert (m["false_matches"], m["missed_matches"], m["latency_p95_s"]) == (1, 1, 3.0), m
assert m["recall_at_k"] == 1.0 and evaluate.score([row(["a", "b"], [], ["c", "a"])], k=1)["recall_at_k"] == 0.0

# end to end: an oracle that also returns the query itself (must be ignored) and crashes once (must be logged)
gold = {q.record.id: q.gold_ids for q in data.load_queries("valid")}
crash_id = next(iter(gold))


def find_matches(query, k=20):
    if query.id == crash_id:
        raise RuntimeError("boom")
    ids = [*gold[query.id], query.id]
    return Result(matches=[Match(i, 1.0, "oracle") for i in ids], candidate_ids=ids, cost_usd=0.0001, tokens=5)


sys.modules["exactmatch.systems.oracle"] = types.SimpleNamespace(find_matches=find_matches, CONFIG={"model": "none"})
with tempfile.TemporaryDirectory() as tmp:
    evaluate.RUNS = Path(tmp)
    out = evaluate.run("oracle", "valid")
    m = json.loads((out / "metrics.json").read_text())
    assert m["errors"] == 1 and m["false_matches"] == 0 and m["recall"] > 0.99 and m["no_match_acc"] == 1.0, m
    assert json.loads((out / "config.json").read_text())["model"] == "none"
    assert sum(1 for _ in open(out / "predictions.jsonl")) == len(gold)
print("ok")
