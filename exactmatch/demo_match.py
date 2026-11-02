"""Live-catalog matcher for the demo. Owner: C.

Rules first: the same model number matches, a different model number does not.
Fireworks is only asked about leftovers that still look similar. Embeddings
(cheap) drop lookalikes before that call, so a lookup spends tokens on the
gray cases and not on WH-1000XM4 vs WH-1000XM5.
"""
import math
import re
from dataclasses import dataclass

from exactmatch import llm
from exactmatch.normalize import model_numbers, normalize_brand, variant_words
from exactmatch.schema import Match, ProductRecord

_STOP = {"the", "and", "for", "with", "from", "over", "into", "your", "this", "that", "you"}


@dataclass
class Row:
    product_id: str
    title: str
    brand: str | None
    price: float | None
    url: str | None
    score: float
    accepted: bool
    confidence: float
    evidence: str
    via: str


def _blob(record: ProductRecord) -> str:
    attrs = record.attrs or {}
    # Title and sku only. Descriptions mention other products and ratings like IP67.
    return " ".join([
        record.model_number or "",
        record.title or "",
        str(attrs.get("mpn") or ""),
        str(attrs.get("sku") or ""),
    ])


def _listed_models(record: ProductRecord) -> list[str]:
    titled = model_numbers(record.title or "")
    return titled or model_numbers(_blob(record))


def _tokens(record: ProductRecord) -> set[str]:
    text = " ".join(part for part in (record.brand, record.title, record.model_number) if part)
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def token_score(query: ProductRecord, item: ProductRecord) -> float:
    a, b = _tokens(query), _tokens(item)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rule(query: ProductRecord, item: ProductRecord) -> tuple[bool, float, str] | None:
    q_models, i_models = _listed_models(query), _listed_models(item)
    q_brand, i_brand = normalize_brand(query.brand), normalize_brand(item.brand)
    shared = set(q_models) & set(i_models)
    if q_brand and i_brand and q_brand != i_brand and not shared:
        return False, 0.9, f"brand {q_brand} conflicts with {i_brand}"
    if not q_models or not i_models:
        q_var, i_var = variant_words(query.title), variant_words(item.title)
        if q_var != i_var:
            extra = ", ".join(sorted(q_var ^ i_var))
            return False, 0.88, f"variant {extra} conflicts"
        return None
    shared = set(q_models) & set(i_models)
    if not shared:
        return False, 0.94, f"model no. {q_models[0]} conflicts with {i_models[0]}"
    q_var, i_var = variant_words(query.title), variant_words(item.title)
    specific = any(len(m) >= 5 for m in shared)
    if q_var != i_var and not specific:
        extra = ", ".join(sorted(q_var ^ i_var))
        return False, 0.9, f"model no. {min(shared, key=len)} conflicts on variant {extra}"
    if q_brand and i_brand and q_brand != i_brand:
        return False, 0.9, f"brand {q_brand} conflicts with {i_brand}"
    shown = max(shared, key=len)
    return True, 0.97, f"model no. {shown} matches"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _doc(record: ProductRecord, kind: str) -> str:
    text = " ".join(part for part in (record.brand, record.title, record.model_number) if part)
    return f"{kind}: {text}"[:1000]


def verify_with_fireworks(query: ProductRecord, item: ProductRecord):
    prompt = (
        "Do these two listings refer to exactly the same product?\n"
        "A different model number, generation, size, bundle, or accessory is not a match.\n"
        'Reply JSON: {"match": true or false, "evidence": "one short sentence"}\n\n'
        f"A: {query.brand or ''} {query.title} {query.model_number or ''}\n"
        f"B: {item.brand or ''} {item.title} {item.model_number or ''}"
    )
    reply = llm.chat(
        llm.SMALL,
        [{"role": "user", "content": prompt}],
        json_mode=True,
        max_tokens=80,
        reasoning_effort="none",
    )
    data = reply.json or {}
    evidence = data.get("evidence") if isinstance(data.get("evidence"), str) else reply.text[:160]
    return bool(data.get("match")), evidence or "Fireworks returned no evidence", reply.cost_usd, reply.tokens


def compare(query: ProductRecord, catalog: list[ProductRecord], *, use_model: bool = True) -> dict:
    others = [item for item in catalog if item.id != query.id and item.url != query.url]
    q_brand = normalize_brand(query.brand)

    def _rank(item: ProductRecord) -> tuple:
        same = bool(q_brand) and normalize_brand(item.brand) == q_brand
        return (same, token_score(query, item))

    # Same brand first, so a near-miss model outranks an unrelated title that shares "wireless".
    baseline = sorted(others, key=_rank, reverse=True)
    vectors: dict[str, list[float]] = {}
    embed_cost = 0.0
    gray_ids = [item.id for item in others if _rule(query, item) is None]
    if use_model and gray_ids:
        texts = [_doc(query, "search_query")] + [_doc(item, "search_document") for item in others]
        vecs, embed_cost = llm.embed(texts)
        vectors = {item.id: vec for item, vec in zip(others, vecs[1:])}
        query_vec = vecs[0]
    else:
        query_vec = None

    cost = embed_cost
    tokens = 0
    calls = 0
    rows: list[Row] = []
    budget = 3
    for item in baseline:
        ruled = _rule(query, item)
        score = token_score(query, item)
        if ruled is not None:
            accepted, confidence, evidence = ruled
            via = "rule"
        else:
            close = True
            if query_vec is not None and item.id in vectors:
                close = _cosine(query_vec, vectors[item.id]) >= 0.55
            if not close:
                accepted, confidence, evidence, via = False, 0.4, "embedding says this is a different product", "embedding"
            elif use_model and budget > 0 and score >= 0.12:
                budget -= 1
                calls += 1
                accepted, evidence, call_cost, call_tokens = verify_with_fireworks(query, item)
                cost += call_cost
                tokens += call_tokens
                confidence = 0.8 if accepted else 0.2
                via = "fireworks"
            else:
                accepted, confidence, evidence, via = False, 0.3, "no shared model number", "unresolved"
        rows.append(Row(
            item.id, item.title, item.brand, item.price, item.url, score,
            accepted, confidence, evidence, via,
        ))
    ours = [row for row in rows if row.accepted]
    ours.sort(key=lambda row: row.confidence, reverse=True)
    rejected = [row for row in rows if not row.accepted and row.via == "rule"]
    rejected.sort(key=lambda row: (
        "model no." in row.evidence and bool(q_brand) and normalize_brand(row.brand) == q_brand,
        bool(q_brand) and normalize_brand(row.brand) == q_brand,
        row.score,
    ), reverse=True)
    return {
        "ours": ours,
        "baseline": rows[:4],
        "rejected": rejected[:3],
        "cost_usd": cost,
        "tokens": tokens,
        "llm_calls": calls,
    }


def as_matches(rows: list[Row]) -> list[Match]:
    return [Match(row.product_id, row.confidence, row.evidence) for row in rows]
