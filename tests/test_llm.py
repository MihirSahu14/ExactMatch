"""Offline check of llm.py (no API key needed). Run: python -m tests.test_llm"""
import os
import tempfile
from pathlib import Path

os.environ["EM_CACHE"] = str(Path(tempfile.mkdtemp()) / "llm.sqlite")  # fresh sqlite cache for this test
from exactmatch import llm  # noqa: E402

calls = []


def fake_post(path, body):
    calls.append(path)
    if path == "embeddings":
        return {"data": [{"index": i, "embedding": [float(i)]} for i in range(len(body["input"]))][::-1],
                "usage": {"prompt_tokens": 1_000_000}}
    return {"choices": [{"message": {"content": 'Sure:\n```json\n{"match": true}\n```'}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}


llm._post = fake_post
if True:
    msgs = [{"role": "user", "content": "same product?"}]
    a = llm.chat(llm.SMALL, msgs)
    b = llm.chat(llm.SMALL, msgs)
    assert a.json == {"match": True} and not a.cached and b.cached and len(calls) == 1
    assert abs(a.cost_usd - 0.25) < 1e-9 and b.cost_usd == a.cost_usd and a.tokens == 2_000_000
    vecs, cost = llm.embed(["x", "y", "z"])
    assert vecs == [[0.0], [1.0], [2.0]] and abs(cost - 0.008) < 1e-9
assert llm.Reply("no json here", 0, 0).json is None
print("ok")
