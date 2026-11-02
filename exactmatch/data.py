"""WDC Products (80% corner cases) -> catalog + queries. See PLAN.md section 1.

Catalog = WDC train offers. Valid/test queries are never in the catalog.
Unseen products (WDC "100% unseen" test file) have no catalog listing, so their gold is empty;
their clusters are split half to valid, half to test so "no match" can be tuned and scored.
cluster_id never leaves this module.
"""
import gzip
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from exactmatch.schema import ProductRecord, Query

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip"
PREFIX = "wdcproductsmulti80cc20rnd"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    zp = RAW / "80multi.zip"
    if not zp.exists():
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(RAW)


def _rows(name):
    path = RAW / f"{PREFIX}{name}.json.gz"
    if not path.exists():
        download()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _clean(v):
    return None if v in (None, "", "None", "nan") else v


def _price(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _record(r):
    return ProductRecord(
        id=f"wdc:{r['id']}",
        title=_clean(r["title"]) or "",
        description=_clean(r["description"]) or "",
        brand=_clean(r["brand"]),
        price=_price(r["price"]),
        currency=_clean(r["priceCurrency"]),
    )


@lru_cache(maxsize=1)
def _load():
    train = _rows("000un_train_large")
    valid = _rows("000un_valid_large")
    seen_test = _rows("000un_gs")
    unseen = _rows("100un_gs")
    valid_unseen = set(sorted({r["cluster_id"] for r in unseen})[::2])
    test_ids = {r["id"] for r in seen_test}  # WDC valid/test share 4 offers; keep them in test only
    splits = {
        "train": train,
        "valid": [r for r in valid if r["id"] not in test_ids] + [r for r in unseen if r["cluster_id"] in valid_unseen],
        "test": seen_test + [r for r in unseen if r["cluster_id"] not in valid_unseen],
    }
    members = defaultdict(set)
    for r in train:
        members[r["cluster_id"]].add(f"wdc:{r['id']}")
    return [_record(r) for r in train], dict(members), splits


def load_catalog() -> list[ProductRecord]:
    return _load()[0]


def load_queries(split: str) -> list[Query]:
    """split: "train" | "valid" | "test". Train queries are catalog items; their gold excludes themselves."""
    _, members, splits = _load()
    return [
        Query(record=_record(r), gold_ids=members.get(r["cluster_id"], set()) - {f"wdc:{r['id']}"})
        for r in splits[split]
    ]


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    catalog = load_catalog()
    catalog_ids = {p.id for p in catalog}
    print(f"catalog: {len(catalog)} listings")
    for split in ("train", "valid", "test"):
        qs = load_queries(split)
        assert all(q.gold_ids <= catalog_ids for q in qs)
        if split != "train":
            assert not any(q.record.id in catalog_ids for q in qs), "query leaked into catalog"
        empty = sum(not q.gold_ids for q in qs)
        avg = sum(len(q.gold_ids) for q in qs) / len(qs)
        print(f"{split}: {len(qs)} queries, {empty} with no match, {avg:.1f} gold per query")
    assert not {q.record.id for q in load_queries("valid")} & {q.record.id for q in load_queries("test")}
    print("ok")
