"""s3: hybrid retrieval to depth 30, model-number rule hints, then ONE listwise call per query.

Rules as hints, not filters: on valid, sharing a 6+ char code is a match only 62% of the time (family codes),
and rejecting on conflicting codes would drop 9% of all gold. So each candidate is annotated ("shares code X",
"different codes: A vs B") and the model decides. Seeing all candidates at once lets it compare near-misses
against each other. One call with ~30 short candidates is cheaper than 20 pointwise calls. Owner: B."""
import os

from exactmatch import llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems import s1
from exactmatch.systems.common import fmt_record

MODEL = os.environ.get("EM_S3_MODEL", s1.MODEL)
DEPTH = int(os.environ.get("EM_S3_DEPTH", "30"))
EFFORT = os.environ.get("EM_S3_EFFORT", "none")  # "none" | "low"
THRESHOLD = float(os.environ.get("EM_S3_THRESHOLD", "0.5"))
CAND_DESC = 150
MAX_TOKENS = 1200 if EFFORT == "none" else 3000

PROMPTS = {
    "listwise": (
        "You match product listings from different web shops. Below is a QUERY listing and {n} numbered CANDIDATE "
        "listings from a catalog. Find every candidate that is EXACTLY the same product as the query: same brand, "
        "same model, same variant (size, capacity, colour, edition, pack size). Wording, language, price and shop "
        "details may differ freely. A different model or part number means a different product. Often several "
        "candidates match; often none do. Candidates that look alike but differ in one detail are the trap: "
        "compare them against each other.\n"
        "Lines starting with 'Codes:' compare model-number-like codes found in the titles; treat them as hints, "
        "not proof (a shared code can be a product family; a missing code proves nothing).\n\n"
        "QUERY:\n{query}\n\n"
        "CANDIDATES:\n{candidates}\n\n"
        "Reply with JSON only: {{\"matches\": [{{\"id\": <candidate number>, \"confidence\": <0 to 1, how sure you "
        "are it is the same product>, \"evidence\": \"<deciding detail, max 12 words>\"}}]}}. "
        "List only candidates you believe match; use an empty list if none do."
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "confidence": {"type": "number"}, "evidence": {"type": "string"}},
        "required": ["id", "confidence", "evidence"]}}},
    "required": ["matches"],
}

CONFIG = {"model": MODEL, "price_per_1m_in_out": llm.price(MODEL), "retrieval": "hybrid rrf(bm25, embed, code)",
          "embed_model": retrieval.EMBED_MODEL, "judge_depth": DEPTH, "verify": "listwise, one call, code hints",
          "threshold": THRESHOLD, "temperature": 0.0, "max_tokens": MAX_TOKENS, "reasoning_effort": EFFORT}


def code_hint(q_codes: set[str], c: ProductRecord) -> str:
    cc = retrieval.model_codes(c.title)
    key = lambda x: (-len(x), x)  # noqa: E731  (stable tie-break: set order varies per process)
    shared = sorted(q_codes & cc, key=key)
    if shared:
        return f"Codes: shares {shared[0].upper()}"
    near = [(a, b) for a in sorted(q_codes, key=key) for b in sorted(cc, key=key)
            if min(len(a), len(b)) >= 5 and (a in b or b in a)]
    if near:
        return f"Codes: {near[0][0].upper()} ~ {near[0][1].upper()} (one contains the other)"
    if q_codes and cc:
        a, b = min(q_codes, key=key), min(cc, key=key)
        return f"Codes: different ({a.upper()} vs {b.upper()})"
    return ""


def prompt(query: ProductRecord, cands: list[ProductRecord]) -> str:
    q_codes = retrieval.model_codes(query.title)
    blocks = []
    for i, c in enumerate(cands, 1):
        hint = code_hint(q_codes, c)
        blocks.append(f"[{i}]\n{fmt_record(c, CAND_DESC)}" + (f"\n{hint}" if hint else ""))
    return PROMPTS["listwise"].format(n=len(cands), query=fmt_record(query), candidates="\n\n".join(blocks))


def listwise(query: ProductRecord, cands: list[ProductRecord], model: str = MODEL, effort: str = EFFORT,
             max_tokens: int = MAX_TOKENS):
    """(dict candidate_id -> (confidence, evidence), completion)."""
    out = llm.chat(prompt(query, cands), model=model, temperature=0.0, max_tokens=max_tokens,
                   json_schema=SCHEMA, reasoning_effort=effort)
    j = llm.parse_json(out.text) or {}
    picked: dict[str, tuple[float, str]] = {}
    for m in j.get("matches") or []:
        try:
            i, conf = int(m.get("id")), min(1.0, max(0.0, float(m.get("confidence", 0))))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= i <= len(cands):
            picked[cands[i - 1].id] = (conf, str(m.get("evidence", ""))[:200])
    return picked, out


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    cands, spent = retrieval.hybrid_search(query, max(k, DEPTH))
    picked, out = listwise(query, cands)
    matches = sorted((Match(pid, round(conf, 3), ev) for pid, (conf, ev) in picked.items() if conf >= THRESHOLD),
                     key=lambda m: -m.confidence)
    return Result(matches=matches, candidate_ids=[c.id for c in cands], cost_usd=spent + out.cost_usd,
                  tokens=out.tokens)
