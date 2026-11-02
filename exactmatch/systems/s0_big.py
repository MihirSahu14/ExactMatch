"""s0_big, the reference: exactly s0, but with the large model. Shows what money alone buys. Owner: B."""
from exactmatch import llm, retrieval
from exactmatch.schema import ProductRecord, Result
from exactmatch.systems.s0 import PROMPTS, config, verify_pointwise  # noqa: F401  (PROMPTS saved with the run)

MODEL = llm.BIG
CONFIG = config(MODEL)


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    return verify_pointwise(query, retrieval.bm25_search(query, k), MODEL)
