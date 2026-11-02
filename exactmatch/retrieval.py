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
        for term in sorted(set(tokens)):  # sorted: set order varies per process, which changed float sums and ties
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
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


# ---------------------------------------------------------------- model-number index (s1+)

_STOP_CODES = {"usb2", "usb3", "1080p", "720p", "4k", "2k", "ddr3", "ddr4", "ddr5", "mp3", "mp4", "h264", "h265",
               "wifi", "5g", "4g", "3g", "ipv6", "ip67", "ip68", "x86", "x64", "a4", "a3"}


def model_codes(text: str) -> set[str]:
    """Model-number-like codes: tokens (or runs joined by - . / space) with both letters and digits, separators
    removed, at least 4 chars. "DS-7204HQHI-K1" -> {"ds7204hqhik1", "7204hqhi"...}; units like "128gb" excluded."""
    out = set()
    text = (text or "").lower()
    for run in re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text):
        parts = [p for p in re.split(r"[-./]", run) if p]
        cands = {"".join(parts)} | set(parts)
        for c in cands:
            if len(c) < 4 or c in _STOP_CODES or not re.search(r"\d", c) or not re.search(r"[a-z]", c):
                continue
            if re.fullmatch(r"\d+(gb|tb|mb|mhz|ghz|hz|mm|cm|ml|in|w|v|mah|kg|g|m|x|p|k|s|th|nd|rd|st)", c):
                continue  # units / ordinals, not codes
            out.add(c)
        if len(parts) == 1 and len(run) >= 6 and run.isdigit():
            out.add(run)  # pure-digit part numbers like 1204071704100
    return out


class CodeIndex:
    def __init__(self, docs: list[tuple[str, set[str]]]):
        self.by_code: dict[str, set[str]] = defaultdict(set)
        for pid, codes in docs:
            for c in codes:
                self.by_code[c].add(pid)
        self.long_codes = sorted(c for c in self.by_code if len(c) >= 6)

    def search(self, codes: set[str], k: int) -> list[tuple[str, float]]:
        """Docs sharing a code (exact = 1.0; one code contains the other, both >= 5 chars = 0.6), weighted by
        rarity so a code shared by 40 offers counts less than one shared by 3."""
        scores: dict[str, float] = defaultdict(float)
        for c in sorted(codes):
            hits = {c: 1.0} if c in self.by_code else {}
            if len(c) >= 5:
                for lc in self.long_codes:
                    if lc != c and (c in lc or (len(lc) >= 5 and lc in c)):
                        hits.setdefault(lc, 0.6)
            for code, w in hits.items():
                ids = self.by_code[code]
                for pid in ids:
                    scores[pid] += w * len(code) / (1 + math.log(len(ids)))
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _codes() -> CodeIndex:
    by_id = catalog_by_id()
    with _lock:
        if "codes" not in _state:
            _state["codes"] = CodeIndex([(pid, model_codes(f"{p.title} {p.model_number or ''}")) for pid, p in by_id.items()])
        return _state["codes"]


def code_search(query: ProductRecord, k: int = 20) -> list[tuple[str, float]]:
    return [(pid, s) for pid, s in _codes().search(model_codes(f"{query.title} {query.model_number or ''}"), k + 1)
            if pid != query.id][:k]


# ---------------------------------------------------------------- embeddings

EMBED_CHARS = 400


# qwen3-embedding-8b beat nomic-embed-text-v1.5 in hybrid recall@20 on valid (0.928 vs 0.912).
EMBED_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"


def _embed_text(p: ProductRecord, role: str) -> str:
    prefix = ("search_query: " if role == "query" else "search_document: ") if "nomic" in EMBED_MODEL else ""
    return prefix + " ".join(filter(None, [p.title, p.brand or "", " ".join((p.description or "").split())]))[:EMBED_CHARS]


def _emb():
    """(ids, unit-normalised matrix) for the catalog; embedded once and cached on disk by llm.embed."""
    import numpy as np
    from exactmatch import llm
    by_id = catalog_by_id()
    with _lock:
        if "emb" not in _state:
            ids = list(by_id)
            vecs, spent = llm.embed([_embed_text(by_id[i], "doc") for i in ids], model=EMBED_MODEL)
            m = np.asarray(vecs, dtype=np.float32)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            _state["emb"] = (ids, m)
            _state["emb_build_cost"] = spent
        return _state["emb"]


def embed_search(query: ProductRecord, k: int = 20) -> tuple[list[tuple[str, float]], float]:
    """Top-k by cosine similarity. Returns (hits, cost_usd of embedding the query; 0 if cached)."""
    import numpy as np
    from exactmatch import llm
    ids, m = _emb()
    (v,), spent = llm.embed([_embed_text(query, "query")], model=EMBED_MODEL)
    v = np.asarray(v, dtype=np.float32)
    sims = m @ (v / (np.linalg.norm(v) + 1e-9))
    top = np.argsort(-sims)[:k + 1]
    return [(ids[i], float(sims[i])) for i in top if ids[i] != query.id][:k], spent


# ---------------------------------------------------------------- hybrid

# Swept on valid: equal weights, RRF k=10 gave recall@10 0.783 / @20 0.928 (BM25 alone: 0.728 / 0.881).
RRF_K = 10
WEIGHTS = {"bm25": 1.0, "embed": 1.0, "code": 1.0}


def hybrid_search(query: ProductRecord, k: int = 20, depth: int = 100) -> tuple[list[ProductRecord], float]:
    """Reciprocal-rank fusion of BM25, embeddings and the model-code index. Returns (top-k records, cost_usd)."""
    by_id = catalog_by_id()
    q_toks = tokenize(record_text(query))
    lists = {"bm25": [pid for pid, _ in _bm25().search(q_toks, depth + 1) if pid != query.id]}
    emb_hits, spent = embed_search(query, depth)
    lists["embed"] = [pid for pid, _ in emb_hits]
    lists["code"] = [pid for pid, _ in code_search(query, depth)]
    fused: dict[str, float] = defaultdict(float)
    for name, ranked in lists.items():
        for r, pid in enumerate(ranked):
            fused[pid] += WEIGHTS[name] / (RRF_K + r + 1)
    top = sorted(fused, key=lambda pid: (-fused[pid], pid))[:k]
    return [by_id[pid] for pid in top], spent
