"""Candidate retrieval over the catalog. Owner: B.

s0 uses plain BM25 (`bm25_search`). Later systems add embeddings and a model-number index on top.
The index is built lazily from data.load_catalog() once per process and shared by all threads.
"""
import math
import re
import threading
from collections import Counter, defaultdict

from exactmatch import data
from exactmatch.schema import ProductRecord

DESC_CHARS = 300  # descriptions are long and noisy; the head carries most of the signal

_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Hyphenated / dotted codes are kept whole *and* split, so
    "KDL-46V5100" yields ["kdl-46v5100", "kdl46v5100", "kdl", "46v5100"] and matches any spelling."""
    out = []
    for tok in _TOKEN.findall((text or "").lower()):
        parts = re.split(r"[-./]", tok)
        out.append(tok)
        if len(parts) > 1:
            out.append("".join(parts))
            out.extend(p for p in parts if p)
    return out


def record_text(p: ProductRecord) -> str:
    return " ".join(filter(None, [p.title, p.title, p.brand or "", (p.description or "")[:DESC_CHARS]]))


class BM25:
    def __init__(self, docs: list[tuple[str, list[str]]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [d for d, _ in docs]
        self.lens = [len(toks) for _, toks in docs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, (_, toks) in enumerate(docs):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, tokens: list[str], k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[i], s) for i, s in top]


_lock = threading.Lock()
_state: dict = {}


def catalog_by_id() -> dict[str, ProductRecord]:
    with _lock:
        if "by_id" not in _state:
            _state["by_id"] = {p.id: p for p in data.load_catalog()}
        return _state["by_id"]


def _bm25() -> BM25:
    by_id = catalog_by_id()
    with _lock:
        if "bm25" not in _state:
            _state["bm25"] = BM25([(pid, tokenize(record_text(p))) for pid, p in by_id.items()])
        return _state["bm25"]


def bm25_search(query: ProductRecord, k: int = 20) -> list[ProductRecord]:
    """Top-k catalog records by BM25 over title (x2) + brand + description head. Never returns the query itself."""
    by_id = catalog_by_id()
    hits = _bm25().search(tokenize(record_text(query)), k + 1)
    return [by_id[pid] for pid, _ in hits if pid != query.id][:k]


def reset() -> None:
    """Drop cached indexes (tests swap the catalog)."""
    with _lock:
        _state.clear()
