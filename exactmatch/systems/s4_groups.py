"""s4_groups: s4, then snap the answer to the catalog's product groups.

The catalog (WDC train offers) comes with train labels saying which catalog listings are the same product
(data.load_queries("train") -> gold_ids per catalog item). A query is one product, so its answer should be one
group: take s4's most confident match and return that listing's whole group, dropping s4 matches outside it.
This fixes the verifier's inconsistency across siblings (it accepts some listings of a product and rejects
others) and removes stray false matches.

CAVEAT for the write-up: on this benchmark the catalog groups ARE how gold is defined, so one correct top match
yields the whole gold set. Real catalogs are often grouped like this, and the labels are train-only, but report
this system separately from s0-s4 and from the Claude baseline, which do not get the groups. Owner: B."""
import threading

from exactmatch import data
from exactmatch.schema import Match, ProductRecord, Result
from exactmatch.systems import s4

PROMPTS = s4.PROMPTS
SIBLING_CONF = 0.9  # confidence given to group members the verifier did not itself accept (times top confidence)
CONFIG = {**s4.CONFIG, "post": "snap to catalog product groups from TRAIN labels: return the top match's group",
          "sibling_conf_factor": SIBLING_CONF}

_lock = threading.Lock()
_groups: dict[str, frozenset[str]] = {}


def groups() -> dict[str, frozenset[str]]:
    """catalog id -> ids of the same product (itself included), from train labels only."""
    with _lock:
        if not _groups:
            for q in data.load_queries("train"):
                _groups[q.record.id] = frozenset(q.gold_ids | {q.record.id})
        return _groups


def snap(res: Result, query_id: str | None = None) -> Result:
    if not res.matches:
        return res
    top = max(res.matches, key=lambda m: (m.confidence, m.product_id))
    grp = groups().get(top.product_id, frozenset({top.product_id}))
    by_id = {m.product_id: m for m in res.matches}
    out = []
    for pid in sorted(grp):
        if pid == query_id:
            continue
        if pid in by_id:
            out.append(by_id[pid])
        else:
            out.append(Match(pid, round(top.confidence * SIBLING_CONF, 3),
                             f"same catalog product as {top.product_id} ({top.evidence})"[:200]))
    out.sort(key=lambda m: -m.confidence)
    return Result(matches=out, candidate_ids=res.candidate_ids, cost_usd=res.cost_usd, tokens=res.tokens)


def find_matches(query: ProductRecord, k: int = 20) -> Result:
    return snap(s4.find_matches(query, k), query.id)
