"""Helpers shared by the systems. Owner: B."""
import os
from concurrent.futures import ThreadPoolExecutor

from exactmatch.schema import ProductRecord

# Parallel LLM calls per query. evaluate.py already runs 8 queries at once, so 8 x 4 = 32 calls in flight.
INNER_WORKERS = int(os.environ.get("EM_INNER_WORKERS", "4"))


def fmt_record(p: ProductRecord, desc_chars: int = 300) -> str:
    lines = [f"Title: {p.title}"]
    if p.brand:
        lines.append(f"Brand: {p.brand}")
    if p.model_number:
        lines.append(f"Model number: {p.model_number}")
    if p.price is not None:
        lines.append(f"Price: {p.price:g} {p.currency or ''}".rstrip())
    if p.description:
        d = " ".join(p.description.split())
        lines.append(f"Description: {d[:desc_chars]}{'...' if len(d) > desc_chars else ''}")
    return "\n".join(lines)


def pmap(fn, items, workers: int = INNER_WORKERS) -> list:
    """Ordered parallel map for per-candidate LLM calls."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
