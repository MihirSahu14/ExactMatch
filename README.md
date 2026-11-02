# ExactMatch

**Paste a product link and get back every listing of that exact product in a catalog. Powered by a small open model that beats an off-the-shelf LLM on accuracy and cost.**

> Build details (shared interfaces, file ownership, timeline) live in [`PLAN.md`](PLAN.md). Where the two disagree, PLAN.md wins.

Marketplaces, price-comparison sites and retail catalog teams keep getting the same product listed under different titles, descriptions and prices by different shops. *"Sony Bravia 46in LCD KDL-46V5100"* and *"Sony 46" BRAVIA V-Series Full HD LCD TV - KDL46V5100"* are the same SKU. A near-miss like the KDL-46**W**5100 is a different product. Getting this wrong causes:

- **Fragmented listings**: reviews, inventory and buy-box get split across copies, which hurts conversion.
- **Wrong price comparisons**: shoppers and pricing teams see the wrong "best price" or compare against the wrong item.
- **Manual review cost**: catalog ops teams pay people to match products by hand.

**What ExactMatch does:**

```
product URL ──► 1. Ingest     fetch page → extract title, brand, model/part no., specs, price
            ──► 2. Retrieve   find top-k candidate listings in the catalog (keyword + embeddings + model-number index)
            ──► 3. Verify     small open model (Fireworks AI) + harness decides which candidates are the EXACT same product
            ──► results       0..n exact matches, each with confidence and evidence (e.g. "model no. KDL46V5100 matches")
```

An empty result is a valid answer: if the product isn't in the catalog, we say so instead of returning the closest lookalike.

---

## How we map to the judging criteria

| Weight | Criterion | How we cover it |
|---|---|---|
| 40% | Benchmark evaluation | **WDC Products** (official splits, hardest "80% corner cases" setting): offers from thousands of web shops with known exact-match groups. Every system runs on the same held-out queries with the same scorer. We report the baseline, each improvement and the gain. |
| 25% | Business value | Link in, matches out: the workflow a catalog or pricing analyst actually runs. We report **match F1, cost per 1k lookups and latency**, and translate the error reduction into analyst-hours saved. |
| 20% | Technical execution | End-to-end pipeline: page ingestion, hybrid retrieval, then LLM verification with normalization, retrieved few-shot examples, rule checks and a confidence-based cascade. Optional LoRA fine-tune trained only on train data. |
| 10% | Presentation & demo | Live: paste a link and get the matching listings with evidence. Show the baseline's wrong answer next to ours. |
| 5% | Documentation | One command reproduces each row of the results table. Every run saves its config, prompts, outputs and metrics. |

---

## Data and benchmark

**Dataset: WDC Products, multi-class, 80% corner cases** (Web Data Commons, University of Mannheim).
- About 11.7k product offers from about 3.3k web shops (2020 Common Crawl, schema.org markup), grouped into about 2.2k real products by `cluster_id`.
- **"Corner cases"** are hard pairs: lookalike products that are different, and matching offers with very different wording. We use the hardest setting (80%) so the baseline has room to fail.
- Landing page: https://webdatacommons.org/largescaleproductcorpus/wdc-products/
- File: https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip (`python -m exactmatch.data --download`)
- Paper: Peeters, Der & Bizer, *WDC Products: A Multi-Dimensional Entity Matching Benchmark* (EDBT 2024).

**How we turn it into a "link in, matches out" benchmark** (implemented in `exactmatch/data.py`):
- **Catalog** = WDC train offers.
- **Queries** = WDC valid offers (for tuning) and WDC test offers (for final scoring). Queries are never in the catalog.
- **Gold** = the catalog offers with the same `cluster_id`.
- WDC's "unseen products" offers have no listing in the catalog, so their gold is empty. Their products are split half to valid and half to test, which scores the "return nothing" behaviour.
- **`cluster_id` never leaves `data.py`**, so no system can see the answer.
- The benchmark files have no live URLs, so the query record goes straight into stage 2. **Stage 1 (ingestion) is evaluated separately** on about 25 hand-labeled real product URLs. Those pages also form the small live catalog used in the demo (`data/live_catalog.jsonl`).

**Rules (non-negotiable):**
1. **Test queries are for final scoring only.** Nobody looks at test errors to tweak prompts.
2. Prompt iteration, threshold tuning and the choice of few-shot examples use **train/valid only**.
3. Few-shot examples and fine-tuning data come from **train only**.
4. Report the number of test queries and how many have no match next to every score.

### Metrics (computed by `exactmatch/evaluate.py`)
- **Primary: match F1**, micro-averaged over (query, returned listing) pairs. Precision means "every returned listing really is the same product", and recall means "we found all of them".
- **Exact-set accuracy**: the share of queries where the returned set equals the gold set exactly. **No-match accuracy**: the share of no-match queries where we correctly return nothing.
- **Retrieval recall@k**: the share of gold matches that make it into the top-k candidates. This is the ceiling for stage 3, and it tells us whether to fix retrieval or verification.
- **Ops**: cost per 1k lookups (Fireworks token usage × price) and mean and p95 latency per lookup.
- **Business translation**: a *missed match* leaves a duplicate live or a competitor price unseen. A *false match* merges two different products or compares the wrong price, which is worse. Both counts are reported.

---

## Systems (the ladder we climb)

Every system takes the same query and candidate pool size k, and returns a `Result` (matches + candidates + cost). Each one is scored on the same test queries, so the gain from each idea is visible.

| ID | System | Idea |
|---|---|---|
| `s0` | **Baseline** | Plain keyword (BM25) top-k, then a small open model asked zero-shot for each candidate: *"Same product? yes/no"*. This is "off the shelf". |
| `s0_big` | **Reference** | Same as `s0` with a large open model. Shows whether our small-model harness matches or beats a model that costs much more. |
| `s1` | **Normalization + better retrieval** | Extract and normalize brand, model/part number, size/capacity and color. Add a model-number index and embeddings to retrieval (hybrid). Use a structured prompt with a matching rubric and JSON output (`{"match": bool, "confidence": 0-1, "evidence": ...}`). |
| `s2` | **+ Retrieved few-shot** | Include the most similar labeled *train* pairs as examples, covering both matches and hard non-matches such as near-miss model numbers, bundles and accessories. |
| `s3` | **+ Rule checks and listwise verification** | Hard signals override the LLM: conflicting model numbers mean **no match**, and the same brand with an exact model number means **match**. The LLM sees all the remaining candidates together so it can tell variants apart. |
| `s4` | **+ Cascade / routing** | Low-confidence decisions go to the large model. Report the share escalated, F1 and the blended cost. |
| `s5` *(optional)* | **+ LoRA fine-tune** | Fine-tune the small model on train pairs (Fireworks fine-tuning) and use it inside `s3`/`s4`. |

**Expected story:** `s0` misses matches that keyword search ranks low and accepts lookalikes. `s1`–`s3` should give the biggest jump. `s4` gives most of the large model's quality at a fraction of its cost.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FIREWORKS_API_KEY=...                                     # never commit this
python -m exactmatch.data --download                             # fetch WDC + sanity-check splits
python -m exactmatch.evaluate --system s0 --split valid --limit 200   # cheap tuning run
python -m exactmatch.evaluate --system s0 --split test                # final score
python -m tests.test_evaluate                                    # scorer self-test
```

Every run writes to `runs/<timestamp>_<system>_<split>/`: `config.json` (system, models, k, git commit), `prompts/`, `predictions.jsonl` (per query: gold, candidates, returned matches with evidence, cost, latency) and `metrics.json`. `results.md` is generated from `runs/` and is never edited by hand. Every LLM call is cached on disk (key = model + prompt + params), so reruns are free.

**Budget:** $50 of Fireworks credits. Per-candidate verification with a small model costs on the order of a dollar per full test run. Large-model and listwise runs cost more or less accordingly. Check spend before any large-model or fine-tuning job.

---

## Repo layout

```
exactmatch/
  schema.py       # ProductRecord, Query, Match, Result: shared, change only with the whole team
  data.py         # WDC → catalog + queries (A)
  evaluate.py     # scorer + run logging (A)
  report.py       # results.md from runs/ (A)
  llm.py          # Fireworks client: retries, JSON parsing, token + cost accounting, disk cache (B)
  retrieval.py    # BM25 + embedding + model-number index over the catalog → top-k candidates (B)
  systems/        # s0.py, s0_big.py, s1.py, ...: find_matches(query, k) -> Result (B)
  ingest.py       # URL → ProductRecord (C)
  normalize.py    # brand / model-number / unit extraction + normalization (C)
  app.py          # demo: paste URL → matches with evidence (C)
tests/
data/             # raw downloads git-ignored; ingest_eval.jsonl + live_catalog.jsonl committed
runs/             # run outputs (commit the final ones)
```

---

## Team of three: who does what

The split follows the pipeline stages, so each person owns separate files. See PLAN.md for the exact interfaces.

| | **A: data, eval and results** (Mihir) | **B: matching engine** | **C: link ingestion, demo and story** |
|---|---|---|---|
| **Owns** | `data.py`, `evaluate.py`, `report.py` | `llm.py`, `retrieval.py`, `systems/` | `schema.py`, `ingest.py`, `normalize.py`, `app.py` |
| **Early** | WDC loader + scorer ✅ | Fireworks client with caching and cost tracking; BM25 retrieval; `s0` + `s0_big` | URL fetch + JSON-LD/meta extraction with LLM fallback |
| **Next** | Score `s0` + `s0_big` on **test**; retrieval recall@k | `s1` hybrid retrieval + structured prompt, then `s2` few-shot | `normalize.py` (model numbers, units, brands) for `s1`/`s3`; label ~25 URLs and build the live catalog |
| **Then** | Error analysis on **valid**, fed back to B and C; final runs | `s3` rules + listwise, then `s4` cascade; start the `s5` fine-tune early if attempting it | Demo app (URL → matches + evidence, baseline vs ours); business numbers |
| **Last hour** | `results.md`, README results section | Freeze systems, help with final runs | Slides + demo script, rehearse |

**Ground rules**
- Branch per person (`a/...`, `b/...`, `c/...`), small PRs into `main` often. Merge `main` into your branch before starting each new system.
- **Only A edits `evaluate.py`** once the baseline is scored.
- Nobody looks at **test** errors. Tune on valid.
- Commit the final `runs/` directories so every number in `results.md` can be traced back to its run.

---

## Timeline (submissions close 4:30 PM sharp)

| By | Milestone |
|---|---|
| 1:15 | `schema.py` on main; `data.py` + scorer working; `s0` scored on test (critical path) |
| 2:00 | `s1` scored; `ingest.py` working on real URLs |
| 2:45 | `s2` + `s3` scored; live catalog + ingestion eval done; demo runs end to end |
| 3:15 | `s4` scored; `s5` only if the fine-tune is already running |
| **3:30** | **Feature freeze.** Final runs, generate `results.md` |
| 4:15 | Demo, slides and README results done; submit |
