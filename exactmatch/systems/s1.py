"""s1: hybrid retrieval (BM25 + qwen3 embeddings + model-number index, fused by RRF) and a structured prompt.
The small model answers each candidate as JSON {match, confidence, evidence}; we keep matches whose confidence
clears THRESHOLD (tuned on valid). Reasoning is off: the answer is short. Owner: B."""
import os

from exactmatch import llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems.common import fmt_record, pmap

# Nemotron (llm.SMALL) stopped answering on serverless on 2026-11-02 (every call hangs); DeepSeek V4.1 Flash with
# reasoning off answers in ~1s. Override with EM_S1_MODEL.
MODEL = os.environ.get("EM_S1_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash")
THRESHOLD = float(os.environ.get("EM_S1_THRESHOLD", "0.5"))  # tuned on valid; cached reruns make sweeps free
MAX_TOKENS = 160

PROMPTS = {
    "verify": (
        "You match product listings from different web shops. Decide whether Listing A and Listing B are offers "
        "for EXACTLY the same product: same brand, same model, same variant (size, capacity, colour, edition, "
        "pack size). Wording, language, price and shop details may differ freely. Model numbers are the strongest "
        "evidence: a different model or part number means a different product, even if everything else is similar. "
        "If a key detail is missing from one listing, judge from what both state.\n\n"
        "Listing A:\n{a}\n\n"
        "Listing B:\n{b}\n\n"
        "Reply with JSON only: {{\"match\": true|false, \"confidence\": <0 to 1, how sure you are they are the same "
        "product>, \"evidence\": \"<the deciding detail, max 15 words>\"}}"
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {"match": {"type": "boolean"}, "confidence": {"type": "number"}, "evidence": {"type": "string"}},
    "required": ["match", "confidence", "evidence"],
}

CONFIG = {"model": MODEL, "price_per_1m_in_out": llm.price(MODEL), "retrieval": "hybrid rrf(bm25, embed, code)",
          "embed_model": retrieval.EMBED_MODEL, "rrf_k": retrieval.RRF_K, "rrf_weights": retrieval.WEIGHTS,
          "verify": "pointwise zero-shot JSON {match, confidence, evidence}", "threshold": THRESHOLD,
          "temperature": 0.0, "max_tokens": MAX_TOKENS, "reasoning_effort": "none"}


def score_json(text: str) -> tuple[float, str]:
    """(probability-like score that the pair matches, evidence). Unparseable output counts as no match."""
    j = llm.parse_json(text) or {}
    try:
        conf = min(1.0, max(0.0, float(j.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    p = conf if j.get("match") is True else (1 - conf if j.get("match") is False else 0.0)
    return p, str(j.get("evidence", ""))[:200]


def judge(query: ProductRecord, c: ProductRecord, model: str = MODEL):
    """(score, evidence, completion) for one candidate."""
    prompt = PROMPTS["verify"].format(a=fmt_record(query), b=fmt_record(c))
    out = llm.chat(prompt, model=model, temperature=0.0, max_tokens=MAX_TOKENS, json_schema=SCHEMA,
                   reasoning_effort="none")
    return score_json(out.text) + (out,)


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    cands, spent = retrieval.hybrid_search(query, k)
    judged = pmap(lambda c: judge(query, c), cands)
    matches = sorted((Match(c.id, round(p, 3), ev) for c, (p, ev, _) in zip(cands, judged) if p >= THRESHOLD),
                     key=lambda m: -m.confidence)
    return Result(matches=matches, candidate_ids=[c.id for c in cands],
                  cost_usd=spent + sum(o.cost_usd for _, _, o in judged),
                  tokens=sum(o.tokens for _, _, o in judged))
