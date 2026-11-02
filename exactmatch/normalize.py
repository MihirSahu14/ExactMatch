"""Brand, model-number, and unit normalization. Owner: C.

`model_number` on a ProductRecord stays empty until `apply` runs. Matching code
should call `model_numbers` itself so KDL-46V5100 and KDL46V5100 compare equal.
"""
import re

from exactmatch.schema import ProductRecord

# Whole words that mark a variant. "S24" vs "S24 Ultra" is not the same SKU.
VARIANTS = {"PRO", "MAX", "ULTRA", "PLUS", "MINI", "LITE", "SE", "FE", "XL"}

# Hyphenated parts (KDL-46V5100, WH-1000XM5) or a letter run plus at least two digits (KDL46V5100, S24).
_MODEL = re.compile(
    r"(?<![\w/])[A-Za-z]{2,}(?:[-/][A-Za-z0-9]{1,})+(?![\w])"
    r"|(?<![\w])[A-Za-z]{1,10}\d{2,}[A-Za-z0-9]{0,10}(?![\w])"
)
# A size is not a model number. Checked after punctuation is stripped.
_SIZE = re.compile(r"^\d+(?:\.\d+)?(?:GB|TB|MB|KB|IN|CM|MM|KG|LB|OZ|HZ|GHZ|MHZ|KHZ|MP|MAH|KW|W|V|K)$")

_SUFFIX = re.compile(
    r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|company|gmbh|electronics|electronic)\b\.?",
    re.I,
)
_ALIASES = {
    "hewlett packard": "hp",
    "international business machines": "ibm",
    "beats by dre": "beats",
    "lg electronics": "lg",
}

_UNIT = re.compile(
    r"(?i)(?<!\w)(\d+(?:\.\d+)?)\s*-?\s*"
    r"(inches|inch|in|cm|mm|gb|gigabytes|gigabyte|tb|terabytes|terabyte|mb|megabytes|kg|lb|lbs|oz|ghz|mhz|hz|mp|mah)\b"
    r"|(\d+(?:\.\d+)?)\""
)
_UNIT_NAME = {
    "inch": "in", "inches": "in", "in": "in",
    "cm": "cm", "mm": "mm",
    "gb": "gb", "gigabyte": "gb", "gigabytes": "gb",
    "tb": "tb", "terabyte": "tb", "terabytes": "tb",
    "mb": "mb", "megabytes": "mb",
    "kg": "kg", "lb": "lb", "lbs": "lb", "oz": "oz",
    "hz": "hz", "ghz": "ghz", "mhz": "mhz",
    "mp": "mp", "mah": "mah",
}


def _canon(token: str) -> str:
    canon = re.sub(r"[^A-Z0-9]", "", token.upper())
    # Drop a short color/finish suffix (WH1000XM4CS, WH1000XM5/B) when the base
    # still ends in a digit and stays long. S24U and KDL46W5100 are unchanged.
    match = re.fullmatch(r"([A-Z0-9]*\d)[A-Z]{1,2}", canon)
    if match and len(match.group(1)) >= 8:
        return match.group(1)
    return canon


def model_numbers(text: str) -> list[str]:
    """Canonical model numbers, longest first. Separators are removed."""
    if not text:
        return []
    spans = [(m.start(), m.end(), m.group()) for m in _MODEL.finditer(text)]
    spans.sort(key=lambda t: (t[1] - t[0]), reverse=True)
    kept: list[tuple[int, int, str]] = []
    for start, end, tok in spans:
        if any(s <= start and end <= e for s, e, _ in kept):
            continue
        kept.append((start, end, tok))
    out: list[str] = []
    seen: set[str] = set()
    for _, _, tok in kept:
        canon = _canon(tok)
        if not 3 <= len(canon) <= 24:
            continue
        if not (re.search(r"[A-Z]", canon) and re.search(r"\d", canon)):
            continue
        if _SIZE.fullmatch(canon) or re.fullmatch(r"IPX?\d{1,2}", canon):
            continue
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    out.sort(key=len, reverse=True)
    return out


def variant_words(text: str) -> set[str]:
    return {w.upper() for w in re.findall(r"[A-Za-z]+", text or "") if w.upper() in VARIANTS}


def normalize_brand(s: str | None) -> str:
    """Lowercase brand key. Empty string if there is nothing to compare."""
    if not s:
        return ""
    text = s.lower().replace("&", " and ")
    text = re.sub(r"[®™©]", "", text)
    text = _SUFFIX.sub(" ", text)
    text = re.sub(r"[^a-z0-9+ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _ALIASES.get(text, text)


def normalize_unit(unit: str) -> str:
    key = unit.lower().strip().rstrip(".")
    return _UNIT_NAME.get(key, key)


def parse_units(text: str) -> list[tuple[float, str]]:
    """Sizes mentioned in text, as (number, canonical unit). '46in' and '46\"' agree."""
    if not text:
        return []
    found: list[tuple[float, str]] = []
    seen: set[tuple[float, str]] = set()
    for m in _UNIT.finditer(text):
        if m.group(3):
            value, unit = float(m.group(3)), "in"
        else:
            value, unit = float(m.group(1)), normalize_unit(m.group(2))
        key = (value, unit)
        if key not in seen:
            seen.add(key)
            found.append(key)
    return found


def apply(record: ProductRecord) -> ProductRecord:
    """Fill model_number from the title, description, mpn, and sku. Brand text is left as extracted."""
    attrs = record.attrs or {}
    blob = " ".join([
        record.title or "",
        record.description or "",
        str(attrs.get("mpn") or ""),
        str(attrs.get("sku") or ""),
    ])
    nums = model_numbers(blob)
    record.model_number = nums[0] if nums else None
    return record
