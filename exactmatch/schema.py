"""Shared interfaces. See PLAN.md section 2; change only with the whole team."""
from dataclasses import dataclass, field


@dataclass
class ProductRecord:
    id: str                          # "wdc:<offer id>" for benchmark, "web:<sha1(url)[:12]>" for live links
    title: str
    description: str = ""
    brand: str | None = None
    price: float | None = None
    currency: str | None = None      # from WDC priceCurrency
    model_number: str | None = None  # empty in raw data; filled by normalize.py
    url: str | None = None           # live links only
    attrs: dict = field(default_factory=dict)  # specs from JSON-LD / meta tags


@dataclass
class Query:
    record: ProductRecord
    gold_ids: set[str]               # catalog ids that truly match; may be empty


@dataclass
class Match:
    product_id: str
    confidence: float                # 0 to 1
    evidence: str                    # e.g. "model no. KDL46V5100 matches"


@dataclass
class Result:                        # one per query, returned by every system
    matches: list[Match]
    candidate_ids: list[str]         # retrieval top-k, used for recall@k
    cost_usd: float = 0.0
    tokens: int = 0
