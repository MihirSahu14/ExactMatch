"""python -m tests.test_normalize"""
from exactmatch.normalize import model_numbers, normalize_brand, parse_units
from exactmatch.schema import ProductRecord
from exactmatch.normalize import apply

a = model_numbers("Sony Bravia 46in LCD KDL-46V5100")
b = model_numbers('Sony 46" BRAVIA V-Series Full HD LCD TV - KDL46V5100')
c = model_numbers("Sony KDL-46W5100 46-inch")
assert a == b == ["KDL46V5100"], a
assert c == ["KDL46W5100"], c
assert "46IN" not in a

xm5 = model_numbers("Sony WH-1000XM5 Wireless Headphones WH1000XM5/B")
xm4 = model_numbers("Sony WH-1000XM4 Black")
assert xm5[0] == "WH1000XM5", xm5
assert xm4 == ["WH1000XM4"], xm4
assert set(xm5).isdisjoint(xm4)
assert model_numbers("WH1000XM4CS") == model_numbers("WH1000XM4CB") == ["WH1000XM4"]
assert model_numbers("waterproof IP67 and IPX8 speaker") == []

assert model_numbers("Samsung Galaxy S24 Ultra 256GB") == ["S24"]
assert "256GB" not in model_numbers("phone 256GB")
assert parse_units("46in screen, 64 GB, 1.5TB") == [(46.0, "in"), (64.0, "gb"), (1.5, "tb")]
assert parse_units('46"') == [(46.0, "in")]
assert normalize_brand("Sony Corporation") == "sony"
assert normalize_brand("Hewlett-Packard Inc.") == "hp"
assert normalize_brand("") == ""

record = ProductRecord(id="web:abc", title="Sony WH-1000XM5", brand="Sony Corporation")
apply(record)
assert record.model_number == "WH1000XM5"
assert record.brand == "Sony Corporation"
print("ok")
