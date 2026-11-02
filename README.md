# ExactMatch

**Paste a product link and get back every listing of that exact product in a catalog. Powered by a small open model that beats an off-the-shelf LLM on accuracy and cost.**

Marketplaces, price-comparison sites and retail catalog teams keep getting the same product listed under different titles, descriptions and prices by different sellers. *"Sony Bravia 46in LCD KDL-46V5100"* and *"Sony 46" BRAVIA V-Series Full HD LCD TV - KDL46V5100"* are the same SKU. A near-miss like the KDL-46**W**5100 is a different product. Getting this wrong causes:

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
| 40% | Benchmark evaluation | Fixed held-out set of **query products with known exact matches** in a public catalog (Abt-Buy). Every system runs on the same queries with the same scorer. We report the baseline, each improvement and the gain. |
| 25% | Business value | Link in, matches out: the workflow a catalog or pricing analyst actually runs. We report **match F1, cost per 1k lookups and latency**, and translate the error reduction into analyst-hours saved. |
| 20% | Technical execution | End-to-end pipeline: page ingestion, hybrid retrieval, then LLM verification with normalization, retrieved few-shot examples, rule checks and a confidence-based cascade. Optional LoRA fine-tune trained only on train queries. |
| 10% | Presentation & demo | Live: paste a link and get the matching listings with evidence. Show the baseline's wrong answer next to ours. |
| 5% | Documentation | One command reproduces each row of the results table. Every run saves its config, prompts, outputs and metrics. |

---

## Data and benchmark

**Dataset: Abt-Buy** (electronics listings from two retailers, with a hand-labeled list of which Abt products match which Buy products). We use the original source files: `Abt.csv`, `Buy.csv` and the perfect-mapping file, from the Leipzig entity-resolution benchmark page or the DeepMatcher/Magellan data repository. Amazon-Google (software) can be added as a second dataset if time allows.

**How we turn it into a "link in, matches out" benchmark:**
- **Catalog** = all Buy listings.
- **Queries** = Abt listings. Each query's gold answer is its set of matching Buy listings, which can be empty.
- **Split by query** with a fixed seed: 60% train, 20% valid, 20% test (about 200+ test queries). The split IDs are committed to the repo.
- The benchmark feeds the query record straight into stage 2, because the dataset's historical listings have no live URLs. **Stage 1 (ingestion) is evaluated separately** on about 25 hand-labeled real product URLs. See "Link ingestion eval" below.

**Rules (non-negotiable):**
1. **The test queries are frozen.** They are used only for final scoring. Nobody looks at test errors to tweak prompts.
2. Prompt iteration, threshold tuning and the choice of few-shot examples use **train/valid only**.
3. The few-shot retrieval index and the fine-tuning data come from **train queries only**.
4. Report the number of test queries and how many have no match next to every score.

### Metrics
- **Primary: match F1**, micro-averaged over (query, returned listing) pairs. Precision means "every returned listing really is the same product", and recall means "we found all of them".
- **Exact-set accuracy**: the share of queries where the returned set equals the gold set exactly, including correctly returning nothing.
- **Retrieval recall@k**: the share of gold matches that make it into the top-k candidates. This is the ceiling for stage 3, and it tells us whether to fix retrieval or verification.
- **Ops**: cost per 1k lookups (Fireworks token usage × price) and mean and p95 latency per lookup.
- **Business translation**: a *missed match* leaves a duplicate live or a competitor price unseen. A *false match* merges two different products or compares the wrong price, which is worse. Report both counts.

### Link ingestion eval
About 25 real product URLs from a few retailers, with title, brand and model number labeled by hand in `data/ingest_eval.jsonl`. Report field-level extraction accuracy. This shows stage 1 works on live pages and is not part of the main 40% benchmark.

---

## Systems (the ladder we climb)

All systems share stage 1 and the same candidate pool size k. They differ in how candidates are retrieved and verified. Each one is scored on the same test queries, so the gain from each idea is visible.

| ID | System | Idea |
|---|---|---|
| `s0` | **Baseline** | Plain keyword (BM25) top-k, then a small open model asked zero-shot for each candidate: *"Same product? yes/no"*. This is "off the shelf". |
| `s0-big` | **Reference** | Same as `s0` with a large open model. Shows whether our small-model harness matches or beats a model that costs much more. |
| `s1` | **Normalization + better retrieval** | Extract and normalize brand, model/part number, size/capacity and color (regex plus the LLM). Add a model-number index and embeddings to retrieval (hybrid). Use a structured prompt with a matching rubric and JSON output (`{"match": bool, "confidence": 0-1, "evidence": ...}`). |
| `s2` | **+ Retrieved few-shot** | Include the most similar labeled *train* pairs as examples, covering both matches and hard non-matches such as near-miss model numbers, bundles and accessories. |
| `s3` | **+ Rule checks and listwise verification** | Hard signals override the LLM: conflicting model numbers mean **no match**, and the same brand with an exact model number means **match**. The LLM sees all the remaining candidates together so it can tell variants apart. |
| `s4` | **+ Cascade / routing** | Low-confidence decisions go to the large model. Report the share escalated, F1 and the blended cost. |
| `s5` *(optional)* | **+ LoRA fine-tune** | Fine-tune the small model on train-query pairs (Fireworks fine-tuning) and use it inside `s3`/`s4`. |

**Expected story:** `s0` misses matches that keyword search ranks low and accepts lookalikes. `s1`–`s3` should give the biggest jump. `s4` gives most of the large model's quality at a fraction of its cost.

---

## Reproducibility
Every run writes to `runs/<timestamp>_<system>_<split>/`:
```
config.json        # system id, model ids, temperature, k, thresholds, git commit
prompts/           # exact prompt templates
predictions.jsonl  # query id, gold ids, candidate ids, returned ids, confidence, evidence, tokens, latency
metrics.json       # match P / R / F1, exact-set acc, recall@k, cost, latency
```
`results.md` is generated from `runs/` and is never edited by hand. Every LLM call is cached (key = model + prompt hash), so reruns are free.

---

## Repo layout (planned)

```
exactmatch/
  schema.py       # ProductRecord, Match: shared by everyone, agreed first
  ingest.py       # URL → ProductRecord (fetch HTML, read schema.org JSON-LD / meta tags, LLM fallback)
  normalize.py    # brand / model-number / unit extraction + normalization
  data.py         # load Abt-Buy → catalog + queries, fixed query split
  llm.py          # Fireworks client: retries, JSON parsing, token + cost accounting, caching
  retrieval.py    # BM25 + embedding + model-number index over the catalog → top-k candidates
  systems/        # s0.py, s1.py, ... each exposes find_matches(query, k) -> list[Match]
  evaluate.py     # scorer (single source of truth) + run logging
  report.py       # builds results.md from runs/
  app.py          # demo: paste URL → matches with evidence
data/             # downloaded datasets (git-ignored), split IDs + ingest_eval.jsonl (committed)
runs/             # run outputs (commit the final ones)
```

**Shared interfaces (agree in the first 10 minutes):**

```python
@dataclass
class ProductRecord:
    id: str
    title: str
    description: str = ""
    brand: str | None = None
    model_number: str | None = None
    price: float | None = None
    url: str | None = None
    attrs: dict = field(default_factory=dict)

@dataclass
class Match:
    product_id: str
    confidence: float
    evidence: str

def ingest(url: str) -> ProductRecord                              # Person C
def find_matches(query: ProductRecord, k: int = 20) -> list[Match]  # Person B (one per system)
```

```bash
python -m exactmatch.evaluate --system s2 --split test   # Person A's harness calls find_matches per query
python -m exactmatch.app                                 # demo
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FIREWORKS_API_KEY=...   # never commit this
python -m exactmatch.data --download
```

**Budget:** $50 of Fireworks credits. About 200 test queries × 20 candidates is roughly 4k verifications per run: cents with a small model and a few dollars with a large one. Listwise verification (one call per query) costs even less. Check spend before any large-model or fine-tuning job.

---

## Team of three: who does what

The split follows the pipeline stages, so each person owns separate files.

| | **A: data, eval and results** | **B: matching engine** | **C: link ingestion, demo and story** |
|---|---|---|---|
| **Owns** | `data.py`, `evaluate.py`, `report.py` | `llm.py`, `retrieval.py`, `systems/` | `ingest.py`, `normalize.py`, `app.py` |
| **First 45 min** | Download Abt-Buy, build catalog + query split, scorer + run logging | Fireworks client with caching and cost tracking; BM25 retrieval; pick the small and large model IDs | `schema.py` (commit first, ~10 min); URL fetch + JSON-LD/meta extraction with LLM fallback |
| **Next** | Score `s0` + `s0-big` on **test** (the critical path); retrieval recall@k | `s1` hybrid retrieval + structured prompt, then `s2` few-shot | `normalize.py` (model numbers, units, brands) for `s1`/`s3`; label ~25 URLs for the ingestion eval |
| **Then** | Error analysis on **valid**, fed back to B and C; 3-seed final runs | `s3` rules + listwise, then `s4` cascade; start the `s5` fine-tune early if attempting it | Demo app (URL → matches + evidence, baseline vs ours); business numbers (cost per 1k lookups, hours saved) |
| **Last hour** | `results.md`, README results section | Freeze systems, help with final runs | Slides + demo script, rehearse |

**Ground rules**
- **C commits `schema.py` in the first 10 minutes.** Everyone builds against it. B and C can use stub data until A's loader lands.
- Branch per person (`a/...`, `b/...`, `c/...`), small PRs merged into `main` often. Pull before starting each new system.
- **Only A edits `evaluate.py` once the baseline is scored.** Changing the scorer mid-day invalidates the comparison.
- Nobody looks at **test** errors. Tune on valid.
- Commit the final `runs/` directories so every number in `results.md` can be traced back to its run.

**Demo catalog note:** Abt-Buy listings are about 15 years old, so a link to today's product page won't have matches in it. For the demo, either use product links whose items are in the catalog (archived retailer pages work well), or add a small catalog of current listings for live links. Keep the benchmark numbers on Abt-Buy.

---

## Timeline (submission deadline 4:30 pm)

| By | Milestone |
|---|---|
| 12:30 | `schema.py` agreed; data loaded; scorer working; `s0` baseline scored on **test**. This is the critical path. |
| 1:30 | `s1` scored; `ingest.py` working on real URLs |
| 2:30 | `s2` + `s3` scored; ingestion eval done; demo app running end-to-end |
| 3:00 | `s4` scored; `s5` only if the fine-tune is already running |
| **3:30** | **Feature freeze.** Final runs with 3 seeds, generate `results.md` |
| 4:15 | Demo, slides and README results section done; submit |
