# ExactMatch: build plan (source of truth)

> **For Claude sessions:** read this file first. Where it disagrees with `README.md`, **this file wins**. Do not change the shared interfaces below without telling the whole team. Only edit files your role owns.

**Product:** paste a product link, get back every listing of that exact product in the catalog. Each match comes with a confidence and evidence. An empty result is a valid answer.

**Changes from README.md:**
1. Dataset is **WDC Products (multi-class)**, not Abt-Buy. Abt-Buy is pairs from 2 shops; WDC has groups (clusters) of the same product across many shops.
2. We use **WDC's official train/valid/test splits**, not our own 60/20/20 split.
3. `find_matches` returns a **`Result`**, not `list[Match]`. The scorer needs the candidates for recall@k, and the cost of each lookup.
4. IDs carry a **source prefix** (`wdc:` / `web:`).
5. New **`Query`** type that holds the gold answers.

---

## 1. Dataset: WDC Products (multi-class, 80% corner cases)

- 11,715 offers from 3,259 web shops (2020 crawl), grouped into 2,162 real products by `cluster_id`. That's about 5 offers per product.
- **"Corner cases" are the hard pairs:** near-miss products that look alike, and matching listings with very different wording. We use the 80% level (the hardest), so the baseline has room to fail.
- Download: `https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip` (about 2.4 MB). It contains gzipped JSON lines; read it with `pd.read_json(path, lines=True)`.
  - Page with all variants: https://webdatacommons.org/largescaleproductcorpus/wdc-products/
- Raw fields per offer: `id, brand, title, description, price, priceCurrency, cluster_id, label`. The multi-class files also have `unseen`.
- There are **no live URLs** in the benchmark files. The 2020 URLs exist only in a 4.3 GB file and are mostly dead, so we don't use them.

### How the benchmark works
*(Implemented in `exactmatch/data.py`; run `python -m exactmatch.data` for the stats and self-check.)*

- **Catalog** = the 2,841 WDC train offers (500 products). Valid and test queries are **never in the catalog**.
- **Gold** for a query = the catalog offers with the same `cluster_id`.
- **"No match" queries:** WDC's "100% unseen" test file has 500 extra products that are not in the catalog, so their gold is empty. Half of those products go to valid and half to test.

| Split | Queries | With no match | Avg gold per query |
|---|---|---|---|
| train | 2,841 (catalog items themselves) | 0 | 6.0 |
| valid | 1,496 | 500 | 3.8 |
| test | 1,500 | 500 | 3.8 |

- **Train queries are catalog items.** Their gold excludes themselves, and **systems must never return `query.id`**.
- **`cluster_id` never leaves `data.py`.** If systems could see it, the answer would leak.
- Few-shot examples and fine-tuning data come from **train only**.

### Live demo (separate from the benchmark)
C builds a small **live catalog**: about 50–100 current listings (a few products × several stores), made by running `ingest.py` on real URLs and saved to `data/live_catalog.jsonl`. These same pages are the ingestion eval (`data/ingest_eval.jsonl`, about 25 URLs with title, brand and model number labeled by hand). The demo runs on the live catalog; the benchmark numbers come from WDC.

---

## 2. Shared interfaces: `exactmatch/schema.py`

This file is on `main` (`exactmatch/schema.py`). Everyone imports it from there.

```python
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
```

### Functions each role exposes

| Owner | Function |
|---|---|
| A | `data.load_catalog() -> list[ProductRecord]` |
| A | `data.load_queries(split: str) -> list[Query]`, where `split` is `"train"`, `"valid"` or `"test"` |
| B | `systems/<id>.py: find_matches(query: ProductRecord, k: int = 20) -> Result` |
| C | `ingest.ingest(url: str) -> ProductRecord` |
| C | `normalize.py`: `model_numbers(text) -> list[str]`, `normalize_brand(s) -> str`, unit/size helpers |

To run the scorer (`exactmatch/evaluate.py`, on `main`):
```bash
python -m exactmatch.evaluate --system s0 --split valid --limit 200   # cheap check: random 200 queries
python -m exactmatch.evaluate --system s0 --split test                # final number only
```
It loads `exactmatch.systems.s0`, calls `find_matches` for each query, times each call, and writes `runs/<timestamp>_<system>_<split>[_nN]/` with `config.json`, `prompts/`, `predictions.jsonl` and `metrics.json`. The default split is `valid`.

**Contract for B's system modules** (`exactmatch/systems/<id>.py`):
- `find_matches(query, k=20) -> Result`, where `Result.candidate_ids` is the retrieval top-k. It is called from **8 threads at once** (`--workers 8`), so it must be thread-safe.
- Optional module-level `CONFIG = {...}` (model ids, temperature, thresholds) is saved into `config.json`.
- Optional module-level `PROMPTS = {"name": template}` dict is saved into `prompts/`.
- If a call raises an exception, the query counts as "no matches" and the traceback is logged under `errors`. The run does not stop.
- Returning `query.id` is ignored by the scorer, but avoid it anyway.

**Metrics** (`metrics.json`): `precision`, `recall`, `f1` (micro-averaged over query × returned listing), `exact_set_acc`, `no_match_acc` (share of no-match queries where we correctly returned nothing), `recall_at_k`, `false_matches`, `missed_matches`, `cost_per_1k_usd`, `tokens`, `latency_mean_s`, `latency_p95_s`, `errors`.

Scorer self-check: `python -m tests.test_evaluate`.

---

## 3. Common code already on `main` (pull before you start)

| File | What it gives you | Check |
|---|---|---|
| `exactmatch/schema.py` | `ProductRecord`, `Query`, `Match`, `Result` | n/a |
| `exactmatch/data.py` | `load_catalog()`, `load_queries(split)`; downloads WDC on first use | `python -m exactmatch.data` |
| `exactmatch/evaluate.py` | the scorer + run logging | `python -m tests.test_evaluate` |
| `exactmatch/report.py` | `python -m exactmatch.report` builds `results.md` from `runs/` | `python -m tests.test_report` |
| `exactmatch/llm.py` | the Fireworks client used by both B and C (see below) | `python -m tests.test_llm` (offline) |
| `requirements.txt`, `.env.example` | `pip install -r requirements.txt`; copy `.env.example` to `.env` and add your key | n/a |

**`llm.py` usage:**
```python
from exactmatch import llm
r = llm.chat(llm.SMALL, [{"role": "user", "content": "..."}], json_mode=True)  # temperature 0 by default
r.json        # parsed dict or None (tolerates code fences / prose)
r.text, r.tokens, r.cost_usd, r.cached
vecs, cost = llm.embed(["search_query: sony kdl46v5100", ...])  # nomic-embed-text-v1.5, batches of 256
```
- **Tested live on 2026-11-02:** chat with both default models, plus embeddings (768-dim).
- **Nemotron and Kimi are reasoning models.** Nemotron thinks for about 500 tokens by default; pass `reasoning_effort="none"` to answer directly (6 tokens instead of 540). An empty answer cut off by `max_tokens` raises an error, so the scorer logs it instead of counting a silent "no match".
- **Thread-safe**, and retries on 429/5xx. It reads `FIREWORKS_API_KEY` from your environment or from `.env`.
- **Disk cache** in `.cache/llm/` (git-ignored). A cache hit returns the *original* cost, so metrics stay identical on reruns while no money is spent.
- **Default models:** `llm.SMALL` = Nemotron 3.5 Lightning 30B-A3B ($0.05 in / $0.20 out per 1M tokens) and `llm.BIG` = Kimi K3 ($3 / $15). B picks the final pair. Any other model must be added to `llm.PRICES`, or its cost falls back to $0.90 per 1M tokens.
- **Putting cost into a `Result`:** sum `r.cost_usd` and `r.tokens` over the calls made for that query.

## 4. Roles and file ownership

| Role | Owns | Branch |
|---|---|---|
| **A: data, eval and results** (Mihir) | `data.py`, `evaluate.py`, `report.py` | `a/data-eval` |
| **B: matching engine** | `retrieval.py`, `systems/`; changes to `llm.py` (shared, so tell C) | `b/...` |
| **C: ingestion, demo and story** | `ingest.py`, `normalize.py`, `app.py` | `c/...` |

- `schema.py` and `data.py` are already on `main`, so build against the real data. Data is not committed; `data.py` downloads it to `data/raw/` on first use.
- The raw data is real web text: about 64% of offers have no brand, about 25% have no description, and prices come in mixed currencies.
- The systems to build (`s0`–`s5`) are the same as in README.md: BM25 + zero-shot → hybrid retrieval + structured prompt → few-shot → rules + listwise → cascade → optional LoRA.

---

## 5. Ground rules
1. **Nobody looks at test errors.** Tune on valid only.
2. **Only A edits `evaluate.py`** once the baseline is scored.
3. Every LLM call is cached (key = model + prompt hash). Never commit `FIREWORKS_API_KEY`. The budget is $50.
4. Work on your own branch. Merge `main` into it before starting each new system, and open small PRs into `main` often.
5. Commit the final `runs/` directories. `results.md` is generated from them, never edited by hand.

---

## 6. Timeline (submissions close 4:30 PM sharp)

| By | Milestone |
|---|---|
| 1:15 | `schema.py` on main; `data.py` + scorer working; `s0` scored on test (critical path) |
| 2:00 | `s1` scored; `ingest.py` working on real URLs |
| 2:45 | `s2` + `s3` scored; live catalog + ingestion eval done; demo runs end to end |
| 3:15 | `s4` scored; `s5` only if the fine-tune is already running |
| **3:30** | **Feature freeze.** Final runs, generate `results.md` |
| 4:15 | Demo, slides and README results done; submit |
