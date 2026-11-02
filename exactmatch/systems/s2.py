"""s2: s1 + few-shot. A fixed set of worked examples mined from TRAIN labels only (data.load_queries("train")):
hard negatives (the BM25 top non-match of a train item: looks alike, different product) and hard positives
(a true match with the least word overlap). Fixed examples keep the prompt prefix identical across calls, so
Fireworks can cache it. Owner: B."""
import os
import random
import threading

from exactmatch import data, llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems import s1
from exactmatch.systems.common import fmt_record, pmap

MODEL = s1.MODEL
THRESHOLD = float(os.environ.get("EM_S2_THRESHOLD", str(s1.THRESHOLD)))
N_POS, N_NEG = 3, 4
SEED = 0
EX_DESC = 120  # short descriptions in examples keep the prompt small

PROMPTS = {
    "system": s1.PROMPTS["verify"].split("Listing A:")[0].strip(),
    "example": "Listing A:\n{a}\n\nListing B:\n{b}",
    "answer": '{{"match": {m}, "confidence": {c}, "evidence": "{e}"}}',
    "verify": "Listing A:\n{a}\n\nListing B:\n{b}\n\nReply with JSON only: {{\"match\": true|false, \"confidence\": "
              "<0 to 1>, \"evidence\": \"<the deciding detail, max 15 words>\"}}",
}

_lock = threading.Lock()
_examples: list = []


def _overlap(a: ProductRecord, b: ProductRecord) -> float:
    ta, tb = set(retrieval.tokenize(a.title)), set(retrieval.tokenize(b.title))
    return len(ta & tb) / max(1, len(ta | tb))


def _evidence(a: ProductRecord, b: ProductRecord, same: bool) -> str:
    ca, cb = retrieval.model_codes(a.title), retrieval.model_codes(b.title)
    if same:
        shared = sorted(ca & cb, key=len, reverse=True)
        return f"same model number {shared[0].upper()}" if shared else "same product, described differently"
    only_a, only_b = sorted(ca - cb, key=len, reverse=True), sorted(cb - ca, key=len, reverse=True)
    if only_a and only_b:
        return f"different model numbers: {only_a[0].upper()} vs {only_b[0].upper()}"
    ta, tb = retrieval.tokenize(a.title), retrieval.tokenize(b.title)
    da = [t for t in dict.fromkeys(ta) if t not in tb][:2]
    db = [t for t in dict.fromkeys(tb) if t not in ta][:2]
    if da or db:
        return f"different variant: {' '.join(da) or '(none)'} vs {' '.join(db) or '(none)'}"
    return "similar listings but a different model or variant"


def examples() -> list[tuple[ProductRecord, ProductRecord, bool, str]]:
    """Mined once per process, deterministic. Uses train labels only."""
    with _lock:
        if _examples:
            return _examples
    by_id = retrieval.catalog_by_id()
    train = [q for q in data.load_queries("train") if q.gold_ids]
    rng = random.Random(SEED)
    pool = rng.sample(train, 400)
    pos, neg = [], []
    for q in pool:
        golds = [by_id[g] for g in q.gold_ids if g in by_id]
        if golds:
            g = min(golds, key=lambda p: _overlap(q.record, p))
            pos.append((_overlap(q.record, g), q.record, g))
        for c in retrieval.bm25_search(q.record, 3):
            if c.id not in q.gold_ids:
                neg.append((_overlap(q.record, c), q.record, c))
                break
    pos.sort(key=lambda t: t[0])                       # least overlap first = hardest positives
    neg.sort(key=lambda t: -t[0])                      # most overlap first = hardest negatives
    # skip near-duplicates of the same pair and anything with empty titles
    chosen_pos = [(a, b, True) for _, a, b in pos if a.title and b.title and _overlap(a, b) > 0.05][:N_POS]
    chosen_neg = [(a, b, False) for _, a, b in neg if a.title and b.title][:N_NEG]
    ex = chosen_pos + chosen_neg
    rng.shuffle(ex)
    out = [(a, b, same, _evidence(a, b, same)) for a, b, same in ex]
    with _lock:
        _examples[:] = out
    return out


def messages(query: ProductRecord, c: ProductRecord) -> list[dict]:
    msgs = [{"role": "system", "content": PROMPTS["system"]}]
    for a, b, same, ev in examples():
        msgs.append({"role": "user", "content": PROMPTS["example"].format(a=fmt_record(a, EX_DESC),
                                                                          b=fmt_record(b, EX_DESC))})
        msgs.append({"role": "assistant", "content": PROMPTS["answer"].format(
            m="true" if same else "false", c=0.95, e=ev)})  # certainty in the answer
    msgs.append({"role": "user", "content": PROMPTS["verify"].format(a=fmt_record(query), b=fmt_record(c))})
    return msgs


def judge(query: ProductRecord, c: ProductRecord, model: str = MODEL):
    out = llm.chat(messages(query, c), model=model, temperature=0.0, max_tokens=s1.MAX_TOKENS,
                   json_schema=s1.SCHEMA, reasoning_effort="none")
    return s1.score_json(out.text) + (out,)


CONFIG = {**s1.CONFIG, "model": MODEL, "price_per_1m_in_out": llm.price(MODEL), "threshold": THRESHOLD,
          "verify": f"pointwise few-shot JSON ({N_POS} hard positives + {N_NEG} hard negatives mined from train)",
          "few_shot_seed": SEED}


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    cands, spent = retrieval.hybrid_search(query, k)
    judged = pmap(lambda c: judge(query, c), cands)
    matches = sorted((Match(c.id, round(p, 3), ev) for c, (p, ev, _) in zip(cands, judged) if p >= THRESHOLD),
                     key=lambda m: -m.confidence)
    return Result(matches=matches, candidate_ids=[c.id for c in cands],
                  cost_usd=spent + sum(o.cost_usd for _, _, o in judged),
                  tokens=sum(o.tokens for _, _, o in judged))
