"""Turn a product URL into a ProductRecord. Owner: C.

Structured data is free, so it wins: JSON-LD Product, then Open Graph and
product meta tags, then the page title. Fireworks (small model, reasoning off)
runs once, and only when title, brand, or price is still missing.

`model_number` stays empty. `normalize.py` fills it.
"""
import hashlib
import ipaddress
import json
import re
import socket
import sys
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

from exactmatch import llm
from exactmatch.schema import ProductRecord

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_BLOCK = ("access denied", "just a moment", "robot check", "attention required", "verify you are human")
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}


@dataclass
class Inspection:
    record: ProductRecord
    sources: dict[str, str]
    title_tag: str
    llm_called: bool
    llm_cost_usd: float = 0.0
    llm_tokens: int = 0
    page_text: str = field(default="", repr=False)


class _Page(HTMLParser):
    """Title, meta tags, JSON-LD blocks, and visible text. Script bodies are not page text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.skip = 0
        self.script_ld = False
        self.script_buf: list[str] = []
        self.ld_json: list[str] = []
        self.metas: list[dict[str, str]] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self.skip += 1
            if tag == "script" and "ld+json" in a.get("type", "").lower():
                self.script_ld = True
                self.script_buf = []
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            self.metas.append(a)
        if tag == "br":
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS and self.skip:
            self.skip -= 1
            if tag == "script" and self.script_ld:
                self.ld_json.append("".join(self.script_buf))
                self.script_ld = False
        if tag == "title":
            self.in_title = False
        if tag in ("p", "div", "li", "h1", "h2", "tr", "section"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.script_ld:
            self.script_buf.append(data)
        elif not self.skip:
            self.text_parts.append(data)


def web_id(url: str) -> str:
    return "web:" + hashlib.sha1(url.encode()).hexdigest()[:12]


def _private_ip(text: str) -> bool:
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    return (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_reserved or addr.is_multicast or addr.is_unspecified
    )


def _local_name(host: str) -> bool:
    return host in {"localhost", "metadata.google.internal"} or host.endswith(".local") or host.endswith(".internal")


def _resolves_private(host: str) -> bool:
    """True when DNS answers with a private address. A name that does not resolve is left to the fetch."""
    if _private_ip(host):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return any(_private_ip(item[4][0]) for item in infos)


def _check_url(url: str, *, resolve: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Use a public http(s) product URL")
    host = parsed.hostname.lower().rstrip(".")
    if _local_name(host) or _private_ip(host) or (resolve and _resolves_private(host)):
        raise ValueError("That host is not a public product page")


def fetch(url: str) -> str:
    """GET the page. Each redirect is checked again so a public URL cannot bounce to a private host."""
    current = url
    session = requests.Session()
    for _ in range(5):
        _check_url(current, resolve=True)
        response = session.get(
            current,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=25,
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                break
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        kind = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if kind and "html" not in kind and "xml" not in kind and not kind.startswith("text/"):
            raise RuntimeError(f"Not an HTML product page ({kind})")
        return response.text[:2_000_000]
    raise RuntimeError("Too many redirects")


def _loads(raw: str):
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _types(node: dict) -> list[str]:
    kind = node.get("@type", [])
    if isinstance(kind, str):
        kind = [kind]
    return [t.split("/")[-1].lower() for t in kind if isinstance(t, str)]


def _is_product(node: dict) -> bool:
    return "product" in _types(node)


def _products(node) -> list[dict]:
    found: list[dict] = []

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if _is_product(value):
                found.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(node)
    return found


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _name(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return _clean(value)
    if isinstance(value, dict):
        return _name(value.get("name"))
    if isinstance(value, list):
        for item in value:
            got = _name(item)
            if got:
                return got
    return None


def _price(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        match = re.search(r"\d+(?:\.\d+)?", cleaned)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _currency(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()[:8]
    return None


def _offer(product: dict) -> tuple[float | None, str | None]:
    offers = product.get("offers")
    if isinstance(offers, list):
        dicts = [item for item in offers if isinstance(item, dict)]
        instock = [item for item in dicts if "instock" in str(item.get("availability", "")).lower()]
        offers = (instock or dicts or [None])[0]
    if not isinstance(offers, dict):
        return None, None
    spec = offers.get("priceSpecification")
    if isinstance(spec, list):
        spec = next((item for item in spec if isinstance(item, dict)), None)
    if not isinstance(spec, dict):
        spec = {}
    price = _price(offers.get("price"))
    if price is None:
        price = _price(offers.get("lowPrice"))
    if price is None:
        price = _price(spec.get("price"))
    currency = _currency(offers.get("priceCurrency")) or _currency(spec.get("priceCurrency"))
    return price, currency


def _attrs(product: dict) -> dict:
    attrs = {}
    for key in ("sku", "mpn", "color", "gtin", "gtin13", "gtin12", "gtin8", "productID"):
        value = product.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
            attrs[key] = str(value).strip()
    for prop in product.get("additionalProperty") or []:
        if not isinstance(prop, dict) or not prop.get("name") or prop.get("value") is None:
            continue
        value = prop["value"]
        attrs[str(prop["name"])] = value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else str(value)
    return attrs


def _meta_map(metas: list[dict[str, str]]) -> dict[str, str]:
    out = {}
    for meta in metas:
        key = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").lower()
        content = meta.get("content") or ""
        if key and content and key not in out:
            out[key] = _clean(content)
    return out


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _overlap(a: str, b: str) -> int:
    return len(_words(a) & _words(b))


def _pick_product(products: list[dict], title_tag: str) -> dict:
    if not products:
        return {}
    if not title_tag:
        return products[0]

    def rank(product: dict) -> tuple:
        price, _ = _offer(product)
        return (_overlap(_name(product.get("name")) or "", title_tag), 1 if price is not None else 0)

    return max(products, key=rank)


def _visible_text(parts: list[str]) -> str:
    text = re.sub(r"[ \t]+", " ", "".join(parts))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:8000]


def _blocked(title_tag: str, page_text: str) -> bool:
    sample = f"{title_tag}\n{page_text[:400]}".lower()
    return any(phrase in sample for phrase in _BLOCK)


def _from_html(url: str, html: str) -> Inspection:
    page = _Page()
    try:
        page.feed(html)
    except Exception:
        pass
    title_tag = _clean("".join(page.title_parts))
    page_text = _visible_text(page.text_parts)
    if _blocked(title_tag, page_text):
        raise RuntimeError(f"The site blocked the fetch ({title_tag[:80]})")

    products: list[dict] = []
    for raw in page.ld_json:
        data = _loads(raw)
        if data is not None:
            products.extend(_products(data))
    product = _pick_product(products, title_tag)
    meta = _meta_map(page.metas)

    json_title = _name(product.get("name"))
    meta_title = meta.get("og:title") or meta.get("twitter:title")
    title = json_title or meta_title
    description = _name(product.get("description")) or meta.get("og:description") or meta.get("description") or ""
    description = re.sub(r"<[^>]+>", " ", description)
    description = _clean(description)[:800]
    json_brand = _name(product.get("brand"))
    brand = json_brand or meta.get("product:brand") or meta.get("og:brand")
    price, currency = _offer(product)
    if price is None:
        price = _price(meta.get("product:price:amount") or meta.get("og:price:amount"))
    if currency is None:
        currency = _currency(meta.get("product:price:currency") or meta.get("og:price:currency"))

    sources: dict[str, str] = {}
    if title:
        sources["title"] = "json-ld" if json_title else "meta"
    if brand:
        sources["brand"] = "json-ld" if json_brand else "meta"
    if price is not None:
        sources["price"] = "json-ld" if _offer(product)[0] is not None else "meta"
    if not title and title_tag:
        title = title_tag
        sources["title"] = "title"

    record = ProductRecord(
        id=web_id(url),
        title=title or "",
        description=description,
        brand=brand,
        price=price,
        currency=currency,
        model_number=None,
        url=url,
        attrs=_attrs(product),
    )
    return Inspection(
        record=record,
        sources=sources,
        title_tag=title_tag,
        llm_called=False,
        page_text=page_text,
    )


def _needs_model(info: Inspection) -> bool:
    record = info.record
    return not record.title or not record.brand or record.price is None


def _fill(info: Inspection) -> Inspection:
    """One cheap Fireworks call. Nemotron reasons for hundreds of tokens unless effort is none."""
    record = info.record
    known = {
        "title": record.title or None,
        "brand": record.brand,
        "description": record.description or None,
        "price": record.price,
        "currency": record.currency,
    }
    prompt = (
        "Extract the single product this page is selling. Ignore nav, ads, and related products.\n"
        "Reply with JSON keys title, brand, description, price, currency, attrs.\n"
        "price is a number or null. currency is a 3-letter code or null. attrs is an object of specs.\n"
        "Use null when the page does not say. Do not invent a model that is not written.\n\n"
        f"URL: {record.url}\nKnown fields: {json.dumps(known)}\n\nPage text:\n{info.page_text[:4000]}"
    )
    reply = llm.chat(
        [{"role": "user", "content": prompt}],
        model=llm.SMALL,
        max_tokens=400,
        json_schema={"type": "object"},
        reasoning_effort="none",
    )
    info.llm_called = True
    info.llm_cost_usd += reply.cost_usd
    info.llm_tokens += reply.tokens
    data = llm.parse_json(reply.text) or {}
    # A raw <title> is usually "Product | Shop". Replace it. Keep JSON-LD and meta titles.
    weak_title = info.sources.get("title") in {None, "title"}
    if (not record.title or weak_title) and _name(data.get("title")):
        record.title = _name(data.get("title")) or ""
        info.sources["title"] = "fireworks"
    if not record.brand and _name(data.get("brand")):
        record.brand = _name(data.get("brand"))
        info.sources["brand"] = "fireworks"
    if not record.description and _name(data.get("description")):
        record.description = (_name(data.get("description")) or "")[:800]
        info.sources["description"] = "fireworks"
    if record.price is None and _price(data.get("price")) is not None:
        record.price = _price(data.get("price"))
        info.sources["price"] = "fireworks"
    if not record.currency and _currency(data.get("currency")):
        record.currency = _currency(data.get("currency"))
        info.sources["currency"] = "fireworks"
    extra = data.get("attrs") if isinstance(data.get("attrs"), dict) else {}
    for key, value in extra.items():
        if key and key not in record.attrs and value not in (None, ""):
            record.attrs[str(key)] = value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else str(value)
    if extra:
        info.sources["attrs"] = "fireworks"
    return info


def inspect(url: str, html: str | None = None) -> Inspection:
    _check_url(url)
    info = _from_html(url, html if html is not None else fetch(url))
    if _needs_model(info) and info.page_text:
        try:
            _fill(info)
        except Exception as exc:
            info.sources["llm_error"] = str(exc)[:180]
    info.record.model_number = None
    return info


def ingest(url: str) -> ProductRecord:
    return inspect(url).record


def _read_urls(path: str) -> list[str]:
    urls = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line.split()[0])
    return urls


def save_catalog(urls: list[str], path: str) -> None:
    from pathlib import Path

    from exactmatch.normalize import apply

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cost, kept = 0.0, 0
    with dest.open("w", encoding="utf-8") as handle:
        for url in urls:
            try:
                info = inspect(url)
            except Exception as exc:
                print(f"FAIL {url} :: {exc}", file=sys.stderr)
                continue
            record = apply(info.record)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            cost += info.llm_cost_usd
            kept += 1
            print(
                f"OK {kept:02d} model={record.model_number or '-':16} "
                f"llm={str(info.llm_called):5} ${info.llm_cost_usd:.4f}  {(record.brand or '')[:16]:16} {record.title[:70]}"
            )
    print(f"saved {kept} listings to {path}  fireworks ${cost:.4f}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest product URLs into ProductRecords")
    parser.add_argument("urls", nargs="*")
    parser.add_argument("--from-file")
    parser.add_argument("--catalog", help="write ProductRecords as jsonl")
    args = parser.parse_args(argv)
    urls = list(args.urls)
    if args.from_file:
        urls.extend(_read_urls(args.from_file))
    if not urls:
        parser.error("pass URLs or --from-file")
    if args.catalog:
        save_catalog(urls, args.catalog)
        return
    for url in urls:
        info = inspect(url)
        print(json.dumps({
            "record": asdict(info.record),
            "sources": info.sources,
            "title_tag": info.title_tag,
            "llm_called": info.llm_called,
            "llm_cost_usd": info.llm_cost_usd,
            "llm_tokens": info.llm_tokens,
        }, indent=2))


if __name__ == "__main__":
    main()
