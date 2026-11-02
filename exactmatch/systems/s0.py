"""s0, the off-the-shelf baseline: BM25 top-k, then a small open model asked zero-shot, one candidate at a time,
"same product? yes/no". Deliberately plain: no normalization, no examples, no rules. Owner: B."""
from exactmatch import llm, retrieval
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems.common import fmt_record, pmap

MODEL = llm.SMALL

PROMPTS = {
    "verify": (
        "Do these two product offers refer to exactly the same product?\n"
        "Answer with only \"yes\" or \"no\".\n\n"
        "Product A:\n{a}\n\n"
        "Product B:\n{b}"
    ),
}


def config(model: str) -> dict:
    return {"model": model, "price_per_1m_in_out": llm.price(model), "retrieval": "bm25",
            "verify": "pointwise zero-shot yes/no", "temperature": 0.0, "max_tokens": 3,
            "reasoning_effort": "none"}


CONFIG = config(MODEL)


def verify_pointwise(query: ProductRecord, candidates: list[ProductRecord], model: str) -> Result:
    def ask(c: ProductRecord):
        prompt = PROMPTS["verify"].format(a=fmt_record(query), b=fmt_record(c))
        # reasoning off: with it on, 3 tokens is all thinking and the answer comes back empty ("no" for everything)
        return llm.chat(prompt, model=model, temperature=0.0, max_tokens=3, reasoning_effort="none")

    outs = pmap(ask, candidates)
    matches = [Match(c.id, 1.0, f"{model.rsplit('/', 1)[-1]} said yes")
               for c, o in zip(candidates, outs) if o.text.strip().lower().startswith("yes")]
    return Result(matches=matches, candidate_ids=[c.id for c in candidates],
                  cost_usd=sum(o.cost_usd for o in outs), tokens=sum(o.tokens for o in outs))


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    return verify_pointwise(query, retrieval.bm25_search(query, k), MODEL)
