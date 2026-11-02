"""Scorer (single source of truth) + run logging. See PLAN.md section 2. Only A edits this file.

    python -m exactmatch.evaluate --system s0 --split valid --limit 200
    python -m exactmatch.evaluate --system s0 --split test

A system is exactmatch/systems/<id>.py with find_matches(query, k) -> Result,
plus optional CONFIG (dict: model ids, temperature, ...) and PROMPTS (dict: name -> template),
both saved with the run.
"""
import argparse
import importlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from exactmatch import data
from exactmatch.schema import Query, Result

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _run_one(find_matches, q: Query, k: int) -> dict:
    t0 = time.perf_counter()
    try:
        res, error = find_matches(q.record, k=k), None
    except Exception:
        res, error = Result(matches=[], candidate_ids=[]), traceback.format_exc(limit=3)
    latency = time.perf_counter() - t0
    matches = {m.product_id: m for m in res.matches if m.product_id != q.record.id}  # never count the query itself
    return {
        "query_id": q.record.id,
        "gold_ids": sorted(q.gold_ids),
        "candidate_ids": list(res.candidate_ids),
        "returned_ids": sorted(matches),
        "matches": [asdict(m) for m in matches.values()],
        "cost_usd": res.cost_usd,
        "tokens": res.tokens,
        "latency_s": latency,
        "error": error,
    }


def score(rows: list[dict], k: int) -> dict:
    tp = fp = fn = exact = cand_hits = 0
    for r in rows:
        gold, got = set(r["gold_ids"]), set(r["returned_ids"])
        tp += len(gold & got)
        fp += len(got - gold)
        fn += len(gold - got)
        exact += gold == got
        cand_hits += len(gold & set(r["candidate_ids"][:k]))
    n = len(rows)
    n_gold = tp + fn
    no_match = [r for r in rows if not r["gold_ids"]]
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / n_gold if n_gold else 0.0
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "queries": n,
        "queries_no_match": len(no_match),
        "precision": p,
        "recall": rc,
        "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "exact_set_acc": exact / n,
        "no_match_acc": sum(not r["returned_ids"] for r in no_match) / len(no_match) if no_match else None,
        "k": k,
        "recall_at_k": cand_hits / n_gold if n_gold else None,
        "false_matches": fp,
        "missed_matches": fn,
        "cost_per_1k_usd": 1000 * sum(r["cost_usd"] for r in rows) / n,
        "tokens": sum(r["tokens"] for r in rows),
        "latency_mean_s": statistics.fmean(lat),
        "latency_p95_s": lat[max(0, math.ceil(0.95 * n) - 1)],
        "errors": sum(r["error"] is not None for r in rows),
    }


def _git_commit() -> str:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def run(system: str, split: str, k: int = 20, limit: int | None = None, workers: int = 8) -> Path:
    mod = importlib.import_module(f"exactmatch.systems.{system}")
    queries = data.load_queries(split)
    if limit:  # random sample, not the first N: test is ordered matched-then-unmatched
        queries = random.Random(0).sample(queries, min(limit, len(queries)))

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(lambda q: _run_one(mod.find_matches, q, k), queries), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"  {i}/{len(queries)}", file=sys.stderr)
    metrics = score(rows, k)

    out = RUNS / (f"{datetime.now():%Y%m%d-%H%M%S}_{system}_{split}" + (f"_n{len(queries)}" if limit else ""))
    (out / "prompts").mkdir(parents=True)
    config = {"system": system, "split": split, "k": k, "limit": limit, "workers": workers,
              "git_commit": _git_commit(), **getattr(mod, "CONFIG", {})}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for name, text in getattr(mod, "PROMPTS", {}).items():
        (out / "prompts" / f"{name}.txt").write_text(text, encoding="utf-8")
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--limit", type=int, help="random sample of N queries (cheap runs)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = run(a.system, a.split, a.k, a.limit, a.workers)
    print((out / "metrics.json").read_text())
    print(f"saved to {out}")
