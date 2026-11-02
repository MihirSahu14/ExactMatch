"""Offline checks for B's code (no network, no API key). Run: python -m tests.test_engine"""
import os
import tempfile
from pathlib import Path

os.environ["EM_CACHE"] = str(Path(tempfile.mkdtemp()) / "llm.sqlite")
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")

from exactmatch import data, llm, retrieval  # noqa: E402
from exactmatch.schema import ProductRecord  # noqa: E402
from exactmatch.systems import s0, s0_big  # noqa: E402

# --- tokenizer keeps model numbers matchable across spellings
toks = retrieval.tokenize("Sony KDL-46V5100 46in")
assert {"kdl-46v5100", "kdl46v5100", "kdl", "46v5100", "46in"} <= set(toks), toks

# --- model-number extraction: codes kept across spellings, units and plain words dropped
mc = retrieval.model_codes
assert "ds7204hqhik1" in mc("Hikvision DS-7204HQHI-K1 Turbo HD DVR"), mc("Hikvision DS-7204HQHI-K1")
assert "tt13ec2s" in mc("Evans TT13EC2S EC2 Clear") and "1204071704100" in mc("Tissot T120 1204071704100")
assert not mc("SSD 128GB 2.5in 1080p 3.9GHz") and not mc("clear drum head")
idx = retrieval.CodeIndex([("a", {"c13t692300"}), ("b", {"t6924"}), ("c", {"xyz123"})])
assert [pid for pid, _ in idx.search({"t692300"}, 5)] == ["a"]  # contained code still hits

# --- fake catalog
CAT = [
    ProductRecord("wdc:1", "Sony Bravia KDL-46V5100 46in LCD TV", brand="Sony", price=999.0, currency="USD"),
    ProductRecord("wdc:2", "Sony KDL46W5100 46 inch Bravia W-series", brand="Sony"),
    ProductRecord("wdc:3", "Samsung Galaxy S20 128GB black", brand="Samsung"),
    ProductRecord("wdc:4", "Logitech MX Master 3 mouse", description="wireless mouse " * 200),
]
data.load_catalog = lambda: CAT
retrieval.reset()
q = ProductRecord("wdc:q", "SONY 46\" BRAVIA V-Series KDL46V5100 Full HD")
cands = retrieval.bm25_search(q, k=3)
assert "wdc:1" in [c.id for c in cands[:2]], [c.id for c in cands]  # plain BM25 may rank the near-miss W5100 first
assert all(c.id != q.id for c in cands)
assert retrieval.bm25_search(CAT[0], k=10)[0].id != "wdc:1", "query itself must be excluded"

# --- fake Fireworks: says yes only if the candidate mentions V5100; fails twice with 429 first
calls = {"n": 0, "fail": 2}


class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._body, self.text = code, body, str(body)

    def json(self):
        return self._body


class FakeSession:
    def post(self, url, json, timeout, headers):
        calls["n"] += 1
        if calls["fail"]:
            calls["fail"] -= 1
            return FakeResp(429, {"error": "rate limited"})
        prompt = json["messages"][-1]["content"]
        b = prompt.split("Product B:")[1]
        ans = "yes" if "v5100" in b.lower() else "no"
        return FakeResp(200, {"choices": [{"message": {"content": ans}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 1}})


llm._session.s = FakeSession()
llm.time.sleep = lambda s: None
s0.pmap = lambda fn, items: [fn(x) for x in items]  # single thread so the fake session (thread-local) is used

res = s0.find_matches(q, k=3)
assert [m.product_id for m in res.matches] == ["wdc:1"], res
n = len(res.candidate_ids)
assert "wdc:1" in res.candidate_ids and 1 <= n <= 3
assert res.tokens == n * 101 and abs(res.cost_usd - llm.cost(llm.SMALL, 100 * n, n)) < 1e-12, res
n_after_first = calls["n"]

# --- cache: same query again makes no HTTP calls but reports the same cost
res2 = s0.find_matches(q, k=3)
assert calls["n"] == n_after_first and res2.cost_usd == res.cost_usd

# --- big model uses a different model id (not served from the small model's cache), priced at its own rate
res3 = s0_big.find_matches(q, k=3)
assert calls["n"] > n_after_first and abs(res3.cost_usd - llm.cost(llm.BIG, 100 * n, n)) < 1e-12
assert s0_big.CONFIG["model"] == llm.BIG and s0.CONFIG["model"] == llm.SMALL

# --- JSON parsing tolerates chatter
assert llm.parse_json('Sure!\n```json\n{"match": true}\n```') == {"match": True}
assert llm.parse_json("no json here") is None

# --- reasoning output is stripped; reasoning budget is added to max_tokens
assert llm.strip_think("<think>hmm, V vs W</think>\nno") == "no"
assert llm.strip_think("<think>cut off by max_tok") == ""
assert all(r["budget"] > 0 for r in llm.REASONING.values() if r["effort"] != "none"), "thinking models need a budget"
print("ok")
