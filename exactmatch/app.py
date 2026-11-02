"""Demo: paste a product URL, get the same listings in the live catalog. Owner: C.

    python -m exactmatch.app

Camera-ready scenes run the real matcher with rules only, so the recording
does not wait on a page fetch or a model call. Pasting a URL still ingests live.
"""
import json
import os
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from exactmatch.demo_match import Row, _rule, compare, token_score
from exactmatch.ingest import inspect
from exactmatch.normalize import apply, model_numbers
from exactmatch.schema import ProductRecord

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "live_catalog.jsonl"
PAGE_FILE = Path(__file__).with_name("demo_page.html")
_cache: dict = {"mtime": None, "rows": []}

SCENES = [
    {
        "id": "xm6",
        "kicker": "Headphones",
        "label": "Sony WH-1000XM6",
        "hook": "The cheaper pair is the previous model.",
        "model": "WH1000XM6",
        "opened": {
            "store": "Sony.com",
            "price": 499.99,
            "title": "Sony WH-1000XM6 Wireless Noise Canceling Headphones",
            "url": "https://electronics.sony.com/audio/headphones/headband-headphones/p/wh1000xm6",
        },
    },
    {
        "id": "bravia65",
        "kicker": "Television",
        "label": "Sony Bravia 65\"",
        "hook": "The 55-inch is a different TV.",
        "model": "K65XR30M2",
        "opened": {
            "store": "Crutchfield",
            "price": 998.0,
            "title": "Sony BRAVIA 3 II TV 65-Inch 4K HDR LED Google TV with Gemini - K65XR30M2",
            "url": "https://www.crutchfield.com/p_158K65XR30/Sony-K65XR30M2.html",
        },
    },
]


def load_catalog() -> list[ProductRecord]:
    mtime = CATALOG.stat().st_mtime if CATALOG.exists() else None
    if _cache["mtime"] != mtime:
        rows = []
        if CATALOG.exists():
            for line in CATALOG.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(apply(ProductRecord(**json.loads(line))))
        _cache["mtime"] = mtime
        _cache["rows"] = rows
    return list(_cache["rows"])


def scene_cards() -> list[dict]:
    cards = []
    for scene in SCENES:
        opened = scene["opened"]
        cards.append({
            "id": scene["id"],
            "kicker": scene["kicker"],
            "label": scene["label"],
            "hook": scene["hook"],
            "store": opened["store"],
            "price": opened["price"],
        })
    return cards


def _anchor(model: str) -> ProductRecord:
    hits = [record for record in load_catalog() if record.model_number == model]
    abt = [record for record in hits if record.url and "abt.com" in record.url]
    if abt:
        return abt[0]
    if hits:
        return hits[0]
    raise KeyError(model)


def _model_prefix(query: ProductRecord, evidence: str) -> int:
    other = evidence.split("conflicts with")[-1].strip()
    best = 0
    blob = " ".join([query.title or "", query.model_number or ""])
    for model in model_numbers(blob):
        n = 0
        for a, b in zip(model, other):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


def _conflicts(query: ProductRecord, catalog: list[ProductRecord]) -> list[Row]:
    """Cheaper listings whose model number is a near miss, closest model first."""
    rows = []
    for item in catalog:
        if item.id == query.id or (query.url and item.url == query.url):
            continue
        ruled = _rule(query, item)
        if not ruled or ruled[0] or not ruled[2].startswith("model no."):
            continue
        accepted, confidence, evidence = ruled
        rows.append(Row(
            item.id, item.title, item.brand, item.price, item.url,
            token_score(query, item), accepted, confidence, evidence, "rule",
        ))
    rows.sort(key=lambda row: (_model_prefix(query, row.evidence), row.score, row.price or 0), reverse=True)
    cheaper = [row for row in rows if row.price is not None and (query.price is None or row.price < query.price)]
    return (cheaper or rows)[:2]


def examples() -> list[dict]:
    ranked = sorted(load_catalog(), key=lambda record: (
        0 if record.model_number in {"WH1000XM4", "WH1000XM5", "WH1000XM6"} else 1,
        record.brand or "",
        record.title or "",
    ))
    seen: set[str] = set()
    out = []
    for record in ranked:
        key = record.model_number or (record.title or "")[:40]
        if not key or key in seen or not record.url:
            continue
        seen.add(key)
        label = record.model_number or record.title[:42]
        if record.brand:
            label = f"{record.brand} {label}"
        out.append({"label": label[:48], "url": record.url, "scene": None})
        if len(out) == 6:
            break
    return out


def _store_name(item: ProductRecord | None, row) -> str:
    attrs = item.attrs if item else {}
    if attrs.get("store"):
        return attrs["store"]
    host = (urlparse((item.url if item else row.url) or "").hostname or "").removeprefix("www.")
    name = host.split(".")[0]
    return name.capitalize() if name else ""


def _row_json(row, by_id: dict) -> dict:
    item = by_id.get(row.product_id)
    return {
        "product_id": row.product_id,
        "title": row.title,
        "brand": row.brand,
        "price": row.price,
        "url": row.url,
        "score": round(row.score, 3),
        "accepted": row.accepted,
        "confidence": row.confidence,
        "evidence": row.evidence,
        "via": row.via,
        "model_hint": (item.model_number if item else "") or "",
        "store": _store_name(item, row),
    }


def _payload(record: ProductRecord, result: dict, *, title_tag: str = "", sources: dict | None = None,
             ingest_llm: bool = False, ingest_cost: float = 0.0, ingest_tokens: int = 0,
             scene: str | None = None) -> dict:
    by_id = {item.id: item for item in load_catalog()}
    return {
        "scene": scene,
        "record": asdict(record),
        "sources": sources or {},
        "title_tag": title_tag,
        "ingest_llm": ingest_llm,
        "cost_usd": round(ingest_cost + result["cost_usd"], 6),
        "tokens": ingest_tokens + result["tokens"],
        "llm_calls": result["llm_calls"] + (1 if ingest_llm else 0),
        "ours": [_row_json(row, by_id) for row in result["ours"]],
        "baseline": [_row_json(row, by_id) for row in result["baseline"]],
        "rejected": [_row_json(row, by_id) for row in result["rejected"]],
    }


def run_scene(scene_id: str) -> dict:
    scene = next((item for item in SCENES if item["id"] == scene_id), None)
    if scene is None:
        raise KeyError(scene_id)
    opened = scene["opened"]
    anchor = _anchor(scene["model"])
    record = apply(ProductRecord(
        id=f"web:scene-{scene_id}",
        title=opened["title"],
        brand=anchor.brand,
        price=opened["price"],
        currency=anchor.currency or "USD",
        url=opened["url"],
        attrs={"store": opened["store"]},
    ))
    catalog = load_catalog()
    result = compare(record, catalog, use_model=False)
    conflicts = _conflicts(record, catalog)
    # Left column is the exact listings plus the model-number near-miss, not a
    # different brand that happens to share the words "wireless headphones".
    result = dict(result)
    result["rejected"] = conflicts
    result["baseline"] = sorted(list(result["ours"]) + conflicts, key=lambda row: row.score, reverse=True)
    return _payload(record, result, scene=scene_id)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, code: int, payload, content_type: str):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/scenes":
            self._send(200, scene_cards(), "application/json")
        elif path == "/api/examples":
            self._send(200, examples(), "application/json")
        else:
            self._send(404, {"error": "Not found"}, "application/json")

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/scene":
                scene_id = (body.get("id") or "").strip()
                if not scene_id:
                    raise ValueError("Unknown scene")
                self._send(200, run_scene(scene_id), "application/json")
                return
            if path != "/api/lookup":
                self._send(404, {"error": "Not found"}, "application/json")
                return
            url = (body.get("url") or "").strip()
            if not url:
                raise ValueError("Paste a product URL")
            info = inspect(url)
            record = apply(info.record)
            result = compare(record, load_catalog(), use_model=True)
            self._send(200, _payload(
                record, result,
                title_tag=info.title_tag,
                sources=info.sources,
                ingest_llm=info.llm_called,
                ingest_cost=info.llm_cost_usd,
                ingest_tokens=info.llm_tokens,
            ), "application/json")
        except KeyError:
            self._send(404, {"error": "Unknown scene"}, "application/json")
        except Exception as exc:
            self._send(400, {"error": str(exc)}, "application/json")


def main():
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ExactMatch demo at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
