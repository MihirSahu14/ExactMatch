"""Fireworks AI client: chat + embeddings with retries, JSON parsing, cost accounting and a disk cache. Owner: B.

    python -m exactmatch.llm --ping      # one tiny call to the small and big model
    python -m exactmatch.llm --models    # list models your key can see

Models are picked with env vars so nobody edits code to switch:
    EM_SMALL_MODEL, EM_BIG_MODEL, EM_EMBED_MODEL

Cached calls return the ORIGINAL usage and cost, so a rerun reports the same cost per 1k lookups as the
first run (the metric is "what this system costs in production", not "what this rerun cost us").
"""
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
# Every chat model on Fireworks serverless (Nov 2026) is a reasoning model. The llama-3.1 / qwen3-4b / maverick
# ids are retired (404). SMALL = 30B MoE with 3B active, thinking off; BIG = gpt-oss-120b, low reasoning.
SMALL = os.environ.get("EM_SMALL_MODEL", "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b")
BIG = os.environ.get("EM_BIG_MODEL", "accounts/fireworks/models/gpt-oss-120b")
EMBED = os.environ.get("EM_EMBED_MODEL", "accounts/fireworks/models/qwen3-embedding-8b")

# USD per 1M tokens (input, output). fireworks.ai/pricing is not reachable from the build container, so these
# are Fireworks' published serverless tiers as we know them -- re-check before quoting cost numbers.
# Unknown models fall back to DEFAULT_PRICE and the run's CONFIG records which price was used.
PRICES: dict[str, tuple[float, float]] = {
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": (0.50, 0.50),  # MoE 0-56B tier
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.60),
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.90, 0.90),           # unverified
    "accounts/fireworks/models/qwen3-embedding-8b": (0.10, 0.0),             # unverified
    "nomic-ai/nomic-embed-text-v1.5": (0.008, 0.0),
}
DEFAULT_PRICE = (0.90, 0.90)

# Per-model reasoning control. `reasoning_effort` is sent to Fireworks; `budget` is extra max_tokens on top of the
# caller's answer budget, because reasoning tokens count against max_tokens (and are billed as output).
# "none" is rejected by gpt-oss and GLM (thinking-only), so they get "low".
REASONING: dict[str, dict] = {
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": {"effort": "none", "budget": 0},
    "accounts/fireworks/models/deepseek-v4p1-flash": {"effort": "none", "budget": 0},
    "accounts/fireworks/models/minimax-m3p1-flash": {"effort": "none", "budget": 0},
    "accounts/fireworks/models/gpt-oss-120b": {"effort": "low", "budget": 512},
    "accounts/fireworks/models/glm-5p3-flash": {"effort": "low", "budget": 512},
}

CACHE_PATH = Path(os.environ.get("EM_CACHE", Path(__file__).resolve().parent.parent / ".cache" / "llm.sqlite"))
TIMEOUT_S = 60
MAX_RETRIES = 6


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool = False

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def price(model: str) -> tuple[float, float]:
    return PRICES.get(model, DEFAULT_PRICE)


def reasoning(model: str) -> dict:
    return REASONING.get(model, {"effort": None, "budget": 0})


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price(model)
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


# ---------------------------------------------------------------- cache (thread-safe sqlite)

class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


_cache = None
_cache_lock = threading.Lock()


def _get_cache() -> _Cache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = _Cache(CACHE_PATH)
        return _cache


def _key(kind: str, payload: dict) -> str:
    return kind + ":" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- HTTP with retries

_session = threading.local()


def _post(path: str, payload: dict) -> dict:
    """POST to Fireworks with exponential backoff on rate limits, server errors and network errors."""
    key = os.environ.get("FIREWORKS_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}  # no key: a network secret may inject it
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
    for attempt in range(MAX_RETRIES):
        try:
            r = _session.s.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT_S, headers=headers)
            if r.status_code == 200:
                return r.json()
            retryable = r.status_code == 429 or r.status_code >= 500
            if r.status_code == 401:
                raise RuntimeError("Fireworks 401: set FIREWORKS_API_KEY (or a network secret for api.fireworks.ai)")
            if not retryable or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Fireworks {r.status_code}: {r.text[:500]}")
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- public API

def chat(messages: list[dict] | str, model: str = SMALL, temperature: float = 0.0, max_tokens: int = 256,
         json_schema: dict | None = None, seed: int | None = None) -> Completion:
    """One chat completion. `messages` may be a plain string (sent as a single user message).

    max_tokens: budget for the ANSWER. The model's reasoning budget (REASONING) is added on top.
    json_schema: if given, asks Fireworks for schema-constrained JSON output. Parse with parse_json().
    seed: vary it to get distinct cached samples at temperature > 0 (self-consistency voting).
    Returned text never contains reasoning: <think>...</think> blocks are stripped.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    r = reasoning(model)
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens + r["budget"]}
    if r["effort"]:
        payload["reasoning_effort"] = r["effort"]
    if json_schema is not None:
        payload["response_format"] = {"type": "json_object", "schema": json_schema}
    if seed is not None:
        payload["seed"] = seed

    cache = _get_cache()
    k = _key("chat", payload)
    hit = cache.get(k)
    if hit is not None:
        return Completion(**hit, cached=True)

    resp = _post("/chat/completions", payload)
    usage = resp.get("usage") or {}
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    out = {"text": strip_think(resp["choices"][0]["message"].get("content") or ""), "prompt_tokens": pt,
           "completion_tokens": ct, "cost_usd": cost(model, pt, ct)}
    cache.put(k, out)
    return Completion(**out)


def embed(texts: list[str], model: str = EMBED, batch_size: int = 128) -> tuple[list[list[float]], float]:
    """Embeddings for texts (cached per text). Returns (vectors, cost_usd of the texts not already cached)."""
    cache = _get_cache()
    keys = [_key("embed", {"model": model, "text": t}) for t in texts]
    vecs: list = [cache.get(k) for k in keys]
    todo = [i for i, v in enumerate(vecs) if v is None]
    spent = 0.0
    for start in range(0, len(todo), batch_size):
        idx = todo[start:start + batch_size]
        resp = _post("/embeddings", {"model": model, "input": [texts[i] for i in idx]})
        for i, item in zip(idx, sorted(resp["data"], key=lambda d: d["index"])):
            vecs[i] = item["embedding"]
            cache.put(keys[i], item["embedding"])
        spent += cost(model, (resp.get("usage") or {}).get("prompt_tokens", 0), 0)
    return vecs, spent


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks (and an unterminated one cut off by max_tokens) from model output."""
    return _THINK_RE.sub("", text or "").strip()


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    """Parse a JSON object from model output; tolerates code fences and chatter around it. None if hopeless."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_RE.search(text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


if __name__ == "__main__":
    if "--models" in sys.argv:
        r = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT_S,
                         headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"} if os.environ.get("FIREWORKS_API_KEY") else {})
        r.raise_for_status()
        for m in sorted(d["id"] for d in r.json().get("data", [])):
            print(m)
    else:
        for m in (SMALL, BIG):
            c = chat("Reply with the single word: pong", model=m, max_tokens=5)
            assert c.text.strip(" .").lower() == "pong", f"{m}: unexpected reply {c.text!r}"
            print(f"{m}: {c.text!r} tokens={c.tokens} cost=${c.cost_usd:.6f} cached={c.cached}")
