"""s4: cascade. s3 (cheap model, one listwise call) answers every query; the big model re-judges the same
candidates only when the cheap answer looks uncertain:
  - any picked candidate has confidence < SURE, or
  - nothing was picked but a candidate shares a model code with the query (likely a missed match), or
  - the cheap output did not parse.
The big model's answer replaces the cheap one for escalated queries. The gate is tuned on valid. Owner: B."""
import os

from exactmatch import llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems import s3

SMALL_MODEL = s3.MODEL
BIG_MODEL = os.environ.get("EM_S4_BIG", llm.BIG)
BIG_EFFORT = os.environ.get("EM_S4_BIG_EFFORT", "none")
SURE = float(os.environ.get("EM_S4_SURE", "0.9"))
THRESHOLD = s3.THRESHOLD
PROMPTS = s3.PROMPTS

CONFIG = {**s3.CONFIG, "model": SMALL_MODEL, "big_model": BIG_MODEL, "big_reasoning_effort": BIG_EFFORT,
          "price_per_1m_in_out": llm.price(SMALL_MODEL), "big_price_per_1m_in_out": llm.price(BIG_MODEL),
          "verify": "cascade: s3 listwise on small model, escalate uncertain queries to big model",
          "escalate_if": f"any pick conf < {SURE}, or no picks but a candidate shares a model code, or parse fail"}


def escalate(query: ProductRecord, cands, picked: dict, text: str) -> str | None:
    """Reason to escalate, or None."""
    if llm.parse_json(text) is None:
        return "parse"
    if any(conf < SURE for conf, _ in picked.values()):
        return "unsure"
    if not picked:
        q_codes = {c for c in retrieval.model_codes(query.title) if len(c) >= 5}
        if any(q_codes & retrieval.model_codes(c.title) for c in cands):
            return "code-but-none"
    return None


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    cands, spent = retrieval.hybrid_search(query, max(k, s3.DEPTH))
    picked, out = s3.listwise(query, cands)
    cost, tokens, tag = spent + out.cost_usd, out.tokens, SMALL_MODEL.rsplit("/", 1)[-1]
    why = escalate(query, cands, picked, out.text)
    if why:
        big_max = 1200 if BIG_EFFORT == "none" else 3000
        picked, out2 = s3.listwise(query, cands, model=BIG_MODEL, effort=BIG_EFFORT, max_tokens=big_max)
        cost, tokens, tag = cost + out2.cost_usd, tokens + out2.tokens, f"{BIG_MODEL.rsplit('/', 1)[-1]} ({why})"
    matches = sorted((Match(pid, round(conf, 3), f"{ev} [{tag}]") for pid, (conf, ev) in picked.items()
                      if conf >= THRESHOLD), key=lambda m: -m.confidence)
    return Result(matches=matches, candidate_ids=[c.id for c in cands], cost_usd=cost, tokens=tokens)
