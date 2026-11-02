"""Shared Fireworks client for systems/ (B) and ingest.py (C): retries, JSON parsing, cost, disk cache.

    from exactmatch import llm
    r = llm.chat(llm.SMALL, [{"role": "user", "content": "..."}], json_mode=True)
    r.json, r.text, r.tokens, r.cost_usd

Thread-safe (the scorer calls systems from 8 threads). Cache key = model + messages + params;
a cache hit returns the ORIGINAL tokens/cost, so run metrics stay the same on reruns (but no money is spent).
"""
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "llm"
API = "https://api.fireworks.ai/inference/v1"

M = "accounts/fireworks/models/"
SMALL = M + "nemotron-lightning-3p5-30b-a3b"   # cheapest listed serverless model; B can change
BIG = M + "kimi-k3"                            # reference "big" model; B can change
EMBED = "nomic-ai/nomic-embed-text-v1.5"

# USD per 1M tokens (input, output), from docs.fireworks.ai/serverless/pricing on 2026-11-02
PRICES = {
    M + "nemotron-lightning-3p5-30b-a3b": (0.05, 0.20),
    M + "glm-5p3-flash": (0.15, 0.50),
    M + "gpt-oss-120b": (0.15, 0.60),
    M + "deepseek-v4p1-flash": (0.30, 1.20),
    M + "minimax-m3": (0.30, 1.20),
    M + "glm-5p3": (1.40, 4.40),
    M + "kimi-k3": (3.00, 15.00),
    EMBED: (0.008, 0.0),
}
FALLBACK_PRICE = (0.90, 0.90)  # ponytail: ">16B" tier as an upper bound for unlisted models; add them to PRICES


def _key():
    if "FIREWORKS_API_KEY" not in os.environ and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("Set FIREWORKS_API_KEY in your environment or in .env (see .env.example)")
    return os.environ["FIREWORKS_API_KEY"]


@dataclass
class Reply:
    text: str
    tokens: int
    cost_usd: float
    cached: bool = False

    @property
    def json(self) -> dict | None:
        """Parsed JSON object from the reply, or None. Tolerates code fences and prose around it."""
        for s in (self.text, *re.findall(r"\{.*\}", self.text, re.S)):
            try:
                v = json.loads(s)
                if isinstance(v, dict):
                    return v
            except ValueError:
                pass
        return None


def _cost(model, prompt_tokens, completion_tokens):
    if model not in PRICES:
        print(f"llm: no price for {model}, using fallback", file=sys.stderr)
    pin, pout = PRICES.get(model, FALLBACK_PRICE)
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


def _post(path, body):
    for attempt in range(6):
        try:
            r = requests.post(f"{API}/{path}", json=body, timeout=90,
                              headers={"Authorization": f"Bearer {_key()}"})
            if r.status_code < 400:
                return r.json()
            if r.status_code not in (408, 429) and r.status_code < 500:
                raise RuntimeError(f"Fireworks {r.status_code}: {r.text[:300]}")
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Fireworks {path}: gave up after retries")


def _cached(body, call):
    path = CACHE / (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    out = call()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{id(out)}.tmp")  # write-then-rename: safe across threads
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, path)
    return out, False


def chat(model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
         json_mode: bool = False, reasoning_effort: str | None = None) -> Reply:
    """reasoning_effort: "none" | "low" | "medium" | "high" for reasoning models (Nemotron, Kimi, ...).
    Nemotron thinks ~500 tokens by default; "none" answers directly."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    def call():
        r = _post("chat/completions", body)
        choice = r["choices"][0]
        text = choice["message"].get("content") or ""
        if not text and choice.get("finish_reason") == "length":  # not cached: raise max_tokens or lower reasoning
            raise RuntimeError(f"{model}: hit max_tokens={max_tokens} before answering (reasoning tokens?)")
        u = r.get("usage", {})
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        # ponytail: ignores Fireworks' cheaper cached-input rate, so cost is a slight overestimate
        return {"text": text, "tokens": pt + ct, "cost_usd": _cost(model, pt, ct)}

    out, hit = _cached(body, call)
    return Reply(out["text"], out["tokens"], out["cost_usd"], hit)


def embed(texts: list[str], model: str = EMBED) -> tuple[list[list[float]], float]:
    """Returns (vectors, cost_usd). Batches of 256. Nomic wants a task prefix, e.g. "search_query: ..."."""
    vecs, cost = [], 0.0
    for i in range(0, len(texts), 256):
        body = {"model": model, "input": texts[i:i + 256]}

        def call():
            r = _post("embeddings", body)
            return {"vecs": [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])],
                    "cost_usd": _cost(model, r.get("usage", {}).get("prompt_tokens", 0), 0)}

        out, _ = _cached(body, call)
        vecs += out["vecs"]
        cost += out["cost_usd"]
    return vecs, cost
