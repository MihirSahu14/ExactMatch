"""Error analysis for one run: where do the misses and false matches come from? VALID runs only.

    python -m exactmatch.errors                      # latest valid run
    python -m exactmatch.errors runs/<run dir>       # a specific valid run

Writes results/errors_<run>.md: counts per bucket plus a few examples each, for B and C to act on.
Refuses test runs: nobody looks at test errors (PLAN.md rule 1).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from exactmatch import data
from exactmatch.evaluate import ROOT, RUNS

# ponytail: crude model-number detector (letters+digits, 4+ chars after stripping -./). Good enough to bucket;
# C's normalize.py is the real one.
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-./]{2,}")


def codes(text: str) -> set[str]:
    out = set()
    for tok in _CODE.findall(text or ""):
        t = re.sub(r"[-./]", "", tok).upper()
        if len(t) >= 4 and re.search(r"[A-Z]", t) and re.search(r"\d", t) and not re.fullmatch(r"\d+[A-Z]{1,3}", t):  # not 46IN / 500GB
            out.add(t)
    return out


def bucket_miss(q, g, retrieved: bool) -> str:
    if not retrieved:
        return "missed: not in retrieval top-k"
    qc, gc = codes(q.title + " " + q.description[:300]), codes(g.title + " " + g.description[:300])
    if qc & gc:
        return "missed: shared model number, verifier said no"
    return "missed: no shared model number (wording differs), verifier said no"


def bucket_false(q, p, query_has_match: bool) -> str:
    if not query_has_match:
        return "false match: product not in catalog, matched a lookalike"
    qc, pc = codes(q.title), codes(p.title)
    if qc and pc and not qc & pc:
        return "false match: model numbers conflict"
    return "false match: lookalike, no model-number conflict (variant / bundle / accessory?)"


def analyze(run_dir: Path, examples: int = 3) -> str:
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg["split"] == "test":
        raise SystemExit("refusing: error analysis on test is not allowed (tune on valid only)")
    records = {p.id: p for p in data.load_catalog()}
    records.update({q.record.id: q.record for q in data.load_queries(cfg["split"])})

    buckets = defaultdict(list)
    for line in open(run_dir / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        q, gold, got, cands = records[r["query_id"]], set(r["gold_ids"]), set(r["returned_ids"]), set(r["candidate_ids"])
        for g in gold - got:
            buckets[bucket_miss(q, records[g], g in cands)].append((q, records[g]))
        for p in got - gold:
            buckets[bucket_false(q, records[p], bool(gold))].append((q, records[p]))
        if r["error"]:
            buckets["errors (exceptions logged by the scorer)"].append((q, None))

    total = sum(len(v) for v in buckets.values())
    out = [f"# Error analysis: `{run_dir.name}`", "",
           f"System `{cfg['system']}`, split `{cfg['split']}`. {total} wrong (query, listing) pairs. "
           "Model-number buckets use a crude regex, so treat them as approximate.", "",
           "| Bucket | Pairs | Share |", "|---|---|---|"]
    for name, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        out.append(f"| {name} | {len(items)} | {len(items) / total:.0%} |")
    for name, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        out += ["", f"## {name}", ""]
        for q, p in items[:examples]:
            out.append(f"- query: `{q.title[:110]}`" + (f"\n  listing: `{p.title[:110]}`" if p else ""))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        valid = [p.parent for p in sorted(RUNS.glob("*/config.json"))
                 if json.loads(p.read_text())["split"] == "valid" and (p.parent / "predictions.jsonl").exists()]
        if not valid:
            raise SystemExit("no valid runs yet")
        run = valid[-1]
    md = analyze(run)
    dest = ROOT / "results" / f"errors_{run.name}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {dest}")
