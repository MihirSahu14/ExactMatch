"""python -m tests.test_demo"""
from exactmatch.demo_match import compare
from exactmatch.normalize import apply
from exactmatch.schema import ProductRecord


def rec(id, title, brand="Sony", url=None):
    record = ProductRecord(id=id, title=title, brand=brand, url=url or f"https://shop.example/{id}")
    return apply(record)


query = rec("web:q", "Sony WH-1000XM5 Wireless Noise Canceling Headphones", url="https://shop.example/query")
catalog = [
    query,
    rec("web:a", "Sony WH-1000XM5 Over-Ear Headphones Black"),
    rec("web:b", "Sony WH-1000XM4 Wireless Noise Canceling Headphones"),
    rec("web:c", "Bose QuietComfort Ultra Headphones", brand="Bose"),
]
result = compare(query, catalog, use_model=False)
ours = {row.product_id for row in result["ours"]}
assert ours == {"web:a"}, ours
assert any(row.product_id == "web:b" and not row.accepted for row in result["baseline"])
assert any("WH1000XM4" in row.evidence for row in result["rejected"])
assert result["llm_calls"] == 0

s24 = rec("web:s", "Samsung Galaxy S24 128GB", brand="Samsung")
ultra = rec("web:u", "Samsung Galaxy S24 Ultra 256GB", brand="Samsung")
plain = rec("web:p", "Samsung Galaxy S24 256GB", brand="Samsung")
result = compare(s24, [s24, ultra, plain], use_model=False)
accepted = {row.product_id for row in result["ours"]}
assert "web:p" in accepted, accepted
assert "web:u" not in accepted, [(row.product_id, row.evidence) for row in result["rejected"]]

from exactmatch.app import run_scene

xm6 = run_scene("xm6")
assert xm6["ours"], xm6
assert all("WH1000XM6" in row["evidence"] for row in xm6["ours"])
assert any("WH1000XM5" in row["evidence"] or "WH1000XM4" in row["evidence"] for row in xm6["rejected"])
assert xm6["llm_calls"] == 0
assert xm6["record"]["attrs"]["store"] == "Sony.com"

bravia = run_scene("bravia65")
assert bravia["ours"], bravia
assert all("K65XR30M2" in row["evidence"] for row in bravia["ours"])
assert any("K55XR30M2" in row["evidence"] or "K43XR30M2" in row["evidence"] for row in bravia["rejected"])
assert all("K55XR30M2" not in row["evidence"] for row in bravia["ours"])
print("ok")
