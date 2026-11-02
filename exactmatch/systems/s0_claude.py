"""s0_claude, the closed-frontier reference: exactly s0 (BM25 top-k, same yes/no prompt, one candidate at a time),
answered by Claude Opus 5.5. Shows how close our open small-model systems get to a frontier closed model.

Run it on the shared sample only (about 6k calls, ~$10-15):
    python -m exactmatch.evaluate --system s0_claude --split test --limit 300

Needs ANTHROPIC_API_KEY (env or .env). Opus 5.5 always thinks; effort "low" keeps it cheap.
temperature is not sent (the API rejects it), so the disk cache is what makes reruns identical.
No refusal fallback on purpose: a silent switch to another model would corrupt the measurement,
so a refusal raises and the scorer logs it as an error. Owner: A.
"""
import os
import threading
from pathlib import Path

import anthropic

from exactmatch import llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems.common import fmt_record, pmap
from exactmatch.systems.s0 import PROMPTS  # noqa: F401  (same prompt as s0; saved with the run)

MODEL = "claude-opus-5-5"
PRICE = (4.00, 20.00)  # USD per 1M tokens (input, output); thinking tokens bill as output
EFFORT = "low"
MAX_TOKENS = 2048      # room for low-effort thinking before the one-word answer

CONFIG = {"model": MODEL, "price_per_1m_in_out": PRICE, "effort": EFFORT, "max_tokens": MAX_TOKENS,
          "retrieval": "bm25", "verify": "pointwise zero-shot yes/no (s0 prompt)"}

_client = None
_lock = threading.Lock()


def _anthropic() -> anthropic.Anthropic:
    global _client
    with _lock:
        if _client is None:
            env = Path(__file__).resolve().parents[2] / ".env"
            if not os.environ.get("ANTHROPIC_API_KEY") and env.exists():
                for line in env.read_text().splitlines():
                    k, _, v = line.partition("=")
                    if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                        os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"')
            _client = anthropic.Anthropic(max_retries=6)  # SDK retries 429/5xx with backoff
        return _client


def ask(prompt: str) -> llm.Completion:
    cache = llm._get_cache()
    key = llm._key("claude", {"model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS, "prompt": prompt})
    hit = cache.get(key)
    if hit is not None:
        return llm.Completion(**hit, cached=True)

    r = _anthropic().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}])
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"{MODEL} stopped with {r.stop_reason}")
    text = next((b.text for b in r.content if b.type == "text"), "")
    pt, ct = r.usage.input_tokens, r.usage.output_tokens
    out = {"text": text, "prompt_tokens": pt, "completion_tokens": ct,
           "cost_usd": (pt * PRICE[0] + ct * PRICE[1]) / 1e6}
    cache.put(key, out)
    return llm.Completion(**out)


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    candidates = retrieval.bm25_search(query, k)
    outs = pmap(lambda c: ask(PROMPTS["verify"].format(a=fmt_record(query), b=fmt_record(c))), candidates)
    matches = [Match(c.id, 1.0, f"{MODEL} said yes")
               for c, o in zip(candidates, outs) if o.text.strip().lower().startswith("yes")]
    return Result(matches=matches, candidate_ids=[c.id for c in candidates],
                  cost_usd=sum(o.cost_usd for o in outs), tokens=sum(o.tokens for o in outs))
