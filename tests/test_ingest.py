"""python -m tests.test_ingest"""
from exactmatch import ingest
from exactmatch.ingest import inspect, web_id

HTML = """
<html><head>
<title>Buy Sony WH-1000XM5 | BigShop</title>
<meta property="og:title" content="Sony WH-1000XM5 Headphones">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product",
 "name":"Sony WH-1000XM5 Wireless Headphones",
 "brand":{"@type":"Brand","name":"Sony"},
 "description":"Noise canceling over-ear headphones",
 "sku":"WH1000XM5/B",
 "offers":{"@type":"Offer","price":"398.00","priceCurrency":"USD"}}
</script>
</head><body><h1>Sony WH-1000XM5</h1><p>Related: WH-1000XM4</p></body></html>
"""

URL = "https://shop.example/wh-1000xm5"
info = inspect(URL, HTML)
assert info.record.id == web_id(URL)
assert info.record.title == "Sony WH-1000XM5 Wireless Headphones"
assert info.record.brand == "Sony"
assert info.record.price == 398.0
assert info.record.currency == "USD"
assert info.record.model_number is None
assert info.record.attrs["sku"] == "WH1000XM5/B"
assert info.llm_called is False
assert info.sources["title"] == "json-ld"

SPARSE = "<html><head><title>Shop</title></head><body><p>Black headphones on sale today.</p></body></html>"
calls = []


def fake_chat(model, messages, temperature=0.0, max_tokens=512, json_mode=False, reasoning_effort=None):
    calls.append({"model": model, "json_mode": json_mode, "reasoning_effort": reasoning_effort, "max_tokens": max_tokens})

    class Reply:
        text = '{"title":"Sony WH-1000XM5","brand":"Sony","description":"Noise canceling","price":348,"currency":"USD","attrs":{"color":"Black"}}'
        tokens = 120
        cost_usd = 0.0002

        @property
        def json(self):
            import json
            return json.loads(self.text)

    return Reply()


ingest.llm.chat = fake_chat
filled = inspect(URL + "/sparse", SPARSE)
assert calls and calls[0]["model"].endswith("nemotron-lightning-3p5-30b-a3b")
assert calls[0]["json_mode"] is True and calls[0]["reasoning_effort"] == "none"
assert filled.llm_called is True
assert filled.record.title == "Sony WH-1000XM5"
assert filled.record.brand == "Sony"
assert filled.record.price == 348
assert filled.record.model_number is None
assert filled.record.attrs["color"] == "Black"
assert filled.llm_cost_usd == 0.0002

try:
    inspect("http://127.0.0.1/secret", "<html></html>")
    raise SystemExit("private host should fail before a model call")
except ValueError:
    pass
for blocked_host in ("http://10.1.2.3/x", "http://localhost/x", "http://192.168.0.4/x"):
    try:
        inspect(blocked_host, "<html><title>Secret</title></html>")
        raise SystemExit(f"{blocked_host} should be refused")
    except ValueError:
        pass

GRAPH = """
<html><head>
<title>Sony WH-1000XM5 Wireless Headphones | BigShop</title>
<script type="application/ld+json">
<!-- page data -->
{"@graph":[
  {"@type":"Product","name":"Sony WH-1000XM4","offers":{"price":"248.00","priceCurrency":"USD"}},
  {"@type":"Product","name":"Sony WH-1000XM5 Wireless Headphones","brand":{"@type":"Brand","name":"Sony"},
   "sku":"WH1000XM5/B",
   "offers":{"@type":"Offer","priceSpecification":{"@type":"UnitPriceSpecification","price":"398.00","priceCurrency":"USD"}}}
]}
</script>
</head><body><p>In stock.</p></body></html>
"""
picked = inspect(URL + "/graph", GRAPH)
assert picked.llm_called is False
assert picked.record.title == "Sony WH-1000XM5 Wireless Headphones"
assert picked.record.price == 398.0
assert picked.record.currency == "USD"
assert picked.record.attrs["sku"] == "WH1000XM5/B"
assert picked.sources["title"] == "json-ld"
assert picked.sources["price"] == "json-ld"

before = len(calls)
try:
    inspect(URL + "/blocked", "<html><head><title>Just a moment...</title></head><body>Checking your browser</body></html>")
    raise SystemExit("a block page should not be ingested")
except RuntimeError:
    pass
assert len(calls) == before
print("ok")
