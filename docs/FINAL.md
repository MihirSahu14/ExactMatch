# ExactMatch: final write-up

*Fireworks: Deploy, Nov 2 2026. Team: Mihir Sahu (data, eval, results), Ishita Kapoor (matching engine) and Harsh Kadodwala (link ingestion and demo). Every number below comes from a run folder committed in `runs/`, and `results.md` is generated from those folders.*

## 1. Summary

**Task.** Given a product listing, or a link to one, return every listing of the **exact same product** in a catalog, with a confidence and evidence for each. If the product isn't in the catalog, return nothing.

**Result.** On a fixed 300-query sample of the test split:
- `s3`, our listwise harness on DeepSeek V4.1 Flash (an open model on Fireworks), reaches **F1 0.863**. Claude Opus 5.5 in the baseline pipeline reaches **F1 0.853**.
- `s3` costs **$0.89 per 1,000 lookups vs $28.21 for Claude, 32× cheaper**, with **2.6× lower p95 latency** (3.7s vs 9.7s).
- `s4` adds a cascade to Kimi K3 for uncertain cases: **F1 0.869** at $3.04 per 1,000 lookups.

**At scale.** On all 1,500 test queries:
- `s3` scores **0.881** and `s4` scores **0.885**, up from the baseline `s0`'s 0.838 on the same model.
- `s3` is also cheaper than that baseline ($0.89 vs $1.12 per 1,000).

**What the comparison means.** Claude ran our *baseline* pipeline (BM25 top 20, one yes/no question per candidate), not our harness. The claim is "a cheap open model inside a good harness beats a frontier model inside a naive one". It does not say the open model is smarter than Claude. Section 9 lists everything we found that weakens the claims, and how we handled it.

## 2. Why this problem

Marketplaces, price-comparison sites and catalog teams get the same product listed many times under different titles. *"Sony Bravia 46in LCD KDL-46V5100"* and *"Sony 46" BRAVIA V-Series Full HD LCD TV - KDL46V5100"* are one SKU. The KDL-46**W**5100 is a different product. Getting this wrong has three costs:
- **Missed matches:** listings stay split, so reviews, inventory and the buy box are split too, and price comparisons miss competitors.
- **False matches:** two different products get merged, or the wrong price is compared. This is worse, because customers see the wrong item.
- **Manual review:** catalog teams pay people to check candidate pairs by hand.

## 3. Choosing the dataset

| Step | Decision | Why |
|---|---|---|
| First plan | **Abt-Buy** (DeepMatcher benchmark) | Well-known and small. |
| Rejected | Abt-Buy | It is **pairs from two shops**, so most products have one match. Our product returns *every* listing of a product, which needs groups of listings. Its listings are also about 15 years old. |
| Chosen | **WDC Products, multi-class, 80% corner cases** ([webdatacommons.org](https://webdatacommons.org/largescaleproductcorpus/wdc-products/)) | Offers from 3,259 web shops (2020 crawl), grouped into real products by `cluster_id`, about 5 offers per product. "80% corner cases" is its hardest setting: near-miss products that look alike, and true matches worded very differently. Official train/valid/test splits. 2.4 MB download. |
| Not used | WDC page URLs | They exist only in a 4.3 GB file and are from 2020, so most are dead. No public benchmark has both live links and labeled product groups, so the benchmark uses records and live links are tested separately (section 8). |

### How we turned it into "one listing in, every match out"
- **Catalog:** the 2,841 WDC **train** offers, covering 500 products.
- **Gold answer** for a query: the catalog offers with the same `cluster_id`. `cluster_id` never leaves `data.py`.
- **Validation and test queries are never in the catalog,** so a system can't find the query itself.
- **"No match" queries:** WDC's "100% unseen" test file has 500 products that aren't in the catalog. We split them by product, half to validation and half to test, so returning nothing is both tuned and scored.

| Split | Queries | With no match | Avg. correct matches per query |
|---|---|---|---|
| validation | 1,496 | 500 | 3.8 |
| test (full) | 1,500 | 500 | 3.8 |
| **test sample** (shared by every system) | 300 | 110 | n/a |

Our first scoring setup (catalog = train + valid + test) would have let queries find themselves, and the WDC "50% unseen" file shares 272 offers with train. The self-check in `python -m exactmatch.data` asserts that neither leak exists.

## 4. Metrics

| Metric | Definition |
|---|---|
| **F1** (primary) | Micro-averaged over (query, returned listing) pairs. Precision = share of returned listings that are truly the same product. Recall = share of true matches found. |
| Exact-set accuracy | Share of queries where the returned set equals the gold set exactly, counting "nothing" as correct for no-match queries. |
| No-match accuracy | Share of no-match queries where the system correctly returned nothing. |
| Recall@20 | Share of true matches that appear in the top 20 retrieved candidates. This is the ceiling for the verifier. |
| Cost per 1,000 lookups | Token usage × list price (Fireworks serverless, Anthropic API). Cached calls report their *original* cost, so reruns don't make a system look free. |
| p95 / mean latency | Wall time per lookup. Only meaningful on a fresh cache; see section 9. |

Error buckets (`python -m exactmatch.errors`, **validation runs only**; it refuses test runs) split every mistake into:
- retrieval misses
- verifier rejections, with and without a shared model number
- false matches on lookalikes or conflicting model numbers

## 5. What we built

| File | Owner | What it does |
|---|---|---|
| `exactmatch/schema.py` | shared | `ProductRecord`, `Query`, `Match`, `Result`: the one contract everyone codes against |
| `exactmatch/data.py` | A | Downloads WDC, builds catalog and splits, self-checks for leaks |
| `exactmatch/evaluate.py` | A | Scorer and run logging. Runs several systems in parallel, 8 queries at a time; a crash is logged per query and never stops the run. Writes `runs/<time>_<system>_<split>/` with config, prompts, predictions and metrics |
| `exactmatch/report.py` | A | Builds `results.md` and 5 charts from `runs/` |
| `exactmatch/errors.py` | A | Error buckets, validation only |
| `exactmatch/llm.py` | B (merged from both) | Fireworks client: per-thread sessions, jittered retries, SQLite cache, cost accounting, `seed`, `json_schema`, `reasoning_effort`, request-sized timeouts, and an error on empty truncated replies |
| `exactmatch/retrieval.py` | B | BM25, Qwen3 embeddings, model-number index, fused with reciprocal rank fusion |
| `exactmatch/systems/` | B (+ A for `s0_claude`) | One file per system, `find_matches(query, k) -> Result` |

`main` got the shared contract first (plan, schema, data loader, scorer). Each person worked on their own branch, merged often, and `main` received everything at the end.

## 6. Models

| Role | Model | Price (USD per 1M tokens, in / out) | Notes |
|---|---|---|---|
| Our verifier | **DeepSeek V4.1 Flash** (Fireworks) | 0.30 / 1.20 | Cheap, but a large MoE model, not a small one. |
| Big open reference / cascade | **Kimi K3** (Fireworks) | 3.00 / 15.00 | Used for `s0_big` and the `s4` escalation. |
| Closed frontier reference | **Claude Opus 5.5** (Anthropic API) | 4.00 / 20.00 | Effort `low`. Thinking can't be turned off on this model, and its answers were about 4 output tokens. |
| Embeddings | **Qwen3 Embedding 8B** (Fireworks) | n/a | Beat nomic-embed-text-v1.5 in hybrid recall@20 on validation (0.928 vs 0.912). |

Lessons from getting models running:
1. **The planned Llama 3.1 IDs return 404.** Fireworks no longer serves them.
2. **Every listed Fireworks model reasons by default.** With `max_tokens=3` they all returned an empty answer, which would have scored the baseline at F1 = 0. The fix is `reasoning_effort="none"`, which GLM 5.3 rejects. We made empty, cut-off replies raise an error instead of silently counting as "no".
3. **Nemotron 3.5 Lightning was our first small model.** It hung on about 1 in 6 requests at 16 at a time, and half its baseline queries failed. Section 9 covers what we did with those runs.

## 7. The systems and how they improved

Each step was motivated by error analysis on the **validation** split.

| System | What it adds | Why (validation evidence) |
|---|---|---|
| `s0` | BM25 top 20, one yes/no call per candidate | Off-the-shelf baseline. |
| `s0_big`, `s0_claude` | Same pipeline, Kimi K3 / Claude Opus 5.5 | What a stronger model alone buys. |
| `s1` | **Hybrid retrieval** (BM25 + embeddings + model-number index), JSON verdict with confidence and evidence | `s0`'s biggest error bucket was retrieval: 44% of mistakes were true matches never retrieved. Recall@20 was capped at 0.869. |
| `s2` | `s1` + few-shot examples mined from **train** labels | The top `s1` bucket (34%) was the verifier rejecting matches written with a different part-number alias, e.g. `WS-C2960-48TC-L` vs `Catalyst 2960 Plus-48TC-L`. |
| `s3` | **One listwise call per query** over 30 candidates, each tagged with a model-number hint ("shares code X" / "different codes") | Comparing candidates against each other separates near-misses, and one call is cheaper than 20. Hints rather than hard rules, because on validation a shared 6+ character code meant a match only 62% of the time, and rejecting on conflicting codes would have dropped 9% of gold. |
| `s4` | **Cascade:** `s3`, then uncertain queries re-judged by Kimi K3 | Uncertain means a pick under 0.9 confidence, no pick although a candidate shares a code, or a reply that doesn't parse. The setting was chosen on validation. |
| `s4_groups` | `s4`, then the whole catalog product group of its top match | **Reported separately;** see section 9. |

### Validation (300-query sample), where tuning happened

| System | F1 | Precision | Recall | Recall@20 | Cost per 1k |
|---|---|---|---|---|---|
| `s0` | 0.836 | 0.920 | 0.766 | 0.869 | $1.11 |
| `s1` | 0.870 | 0.931 | 0.817 | 0.922 | $3.09 |
| `s2` | 0.872 | 0.933 | 0.818 | 0.925 | $8.02 |
| `s3` | 0.879 | 0.941 | 0.825 | 0.923 | **$0.88** |
| `s4` | **0.888** | 0.941 | 0.841 | 0.923 | $3.16 |

`s1` cut validation retrieval misses from 144 to 84.

### Test, shared 300-query sample (final runs on an empty cache)

| System | F1 | Precision | Recall | No-match acc | Exact-set acc | Cost per 1k | p95 latency |
|---|---|---|---|---|---|---|---|
| `s0` (DeepSeek V4.1 Flash) | 0.814 | 0.880 | 0.757 | 0.982 | 0.573 | $1.13 | 26.5s* |
| `s0_big` (Kimi K3) | 0.824 | 0.920 | 0.746 | 0.982 | 0.597 | $12.50 | 10.6s |
| **`s0_claude` (Claude Opus 5.5)** | **0.853** | 0.944 | 0.778 | 0.991 | 0.653 | **$28.21** | 9.7s |
| `s1` | 0.853 | 0.886 | 0.822 | 0.973 | 0.630 | $3.11 | 35.6s* |
| `s2` | 0.783 | 0.889 | 0.699 | 0.964 | 0.527 | $8.03 | 21.0s* |
| **`s3`** | **0.863** | 0.927 | 0.808 | 0.991 | **0.683** | **$0.89** | **3.7s** |
| **`s4`** | **0.869** | 0.934 | 0.812 | 0.991 | 0.677 | $3.04 | 7.7s |

\* The five systems ran side by side, which inflates latency for the per-candidate systems (20 calls each).

### Test, full split (1,500 queries)

| System | F1 | Precision | Recall | No-match acc | Exact-set acc | Cost per 1k |
|---|---|---|---|---|---|---|
| `s0` | 0.838 | 0.913 | 0.774 | 0.968 | 0.587 | $1.12 |
| `s1` | 0.871 | 0.918 | 0.829 | 0.960 | 0.665 | $3.10 |
| `s3` | 0.881 | 0.944 | 0.826 | 0.974 | 0.680 | $0.89 |
| `s4` | **0.885** | 0.947 | 0.830 | 0.978 | 0.689 | $3.07 |
| *`s4_groups` (see section 9)* | *0.959* | *0.968* | *0.951* | *0.978* | *0.955* | *$3.07* |

Claude was run on the 300-query sample only; we chose not to spend about $42 and an hour running it on all 1,500.

### Cost per 1,000 lookups

| System | Cost | vs Claude | Why |
|---|---|---|---|
| `s0_claude` | $28.21 | n/a | 20 calls per lookup on a $4 / $20 model |
| `s0_big` | $12.50 | 2.3× cheaper | 20 calls per lookup on Kimi K3 |
| `s2` | $8.03 | 3.5× cheaper | Few-shot examples added to each of 20 prompts |
| `s1` | $3.11 | 9× cheaper | 20 JSON calls per lookup |
| `s4` | $3.04 | 9× cheaper | `s3` plus Kimi K3 on uncertain queries |
| `s0` | $1.13 | 25× cheaper | 20 short yes/no calls |
| **`s3`** | **$0.89** | **32× cheaper** | **One** listwise call per lookup |

## 8. Live link check

We pasted a real Best Buy link: the [Keychron V1 Max](https://www.bestbuy.com/product/keychron-v1-max-75-rgb-mechanical-keyboard-qmk-built-in-ai-launcher-hot-swappable-black/CGC7F9C276/sku/13005877), model `V1M-D4`, $114.99. We read the page's built-in product data (schema.org JSON-LD) in a browser.
- **Result:** `s0`, `s1`, `s2` and `s0_claude` all correctly returned **no match**. The 2020 catalog has no keyboards; the nearest candidates were Razer mice, a printer and a UPS battery.
- **Speed and cost:** `s1` took 10.5s and cost $0.003; Claude cost $0.036.
- **Limit:** "no match" is the right answer, but a positive live demo needs a small catalog of current listings, which we did not build.

## 9. Honesty: problems we found, and when

None of these came from code reading test labels. We audited the code at the end: no system reads validation or test labels, only `s2` and `s4_groups` use labels at all (train only), and retrieval indexes only the catalog. The process problems below are real, and we report them as they happened.

1. **We looked at the test split many times, not once.** Our plan said the test split is frozen and used only for final scoring. In practice we ran systems on the test sample from 13:44 onwards, 17 test runs in all, while `s1`–`s4` and `s4_groups` were still being built. Nobody read test *errors*, and every threshold was tuned on validation. But design choices after 14:49 were made with test scores visible, and `s4_groups` was built after full-test `s3`/`s4` scores were known. *Caught at the final audit.* **Mitigation:** the validation ladder shows the same order of systems as test.
2. **`s4_groups` uses the same groups that define the correct answers.** It returns the whole catalog group of the top match, using train labels. That's a fair production design when a catalog is already grouped into products. But in this benchmark, gold *is* those groups, so one correct top match yields the entire answer. Claude and `s0`–`s4` never got the groups. *Caught when it landed:* it is reported only in its own row and section, never in the comparison with Claude.
3. **The first small-model baseline measured an outage, not the model.** Nemotron runs had read timeouts on 150 of 300 test queries and 133 of 300 validation queries, giving F1 of about 0.30. *Caught when the runs finished:* they are moved to `runs/_invalid/` with a note, and we switched to DeepSeek V4.1 Flash.
4. **Cached reruns reported fake latency.** A rerun served from the LLM cache showed p95 latency near 0.1s. *Caught mid-afternoon:* the final 300-query test runs used an empty cache. Ishita's full-split runs reused cached calls, so **their latency is not reliable** (e.g. `s4_groups` 0.1s) and we don't quote it. Costs are unaffected, because the cache stores each call's original cost.
5. **The comparison with Claude is not like for like.** Claude ran our baseline pipeline, not our harness, at effort `low`. A Claude inside the `s3`/`s4` harness might well score higher. Our claim is about harness design and cost, not raw model ability.
6. **`s2` regressed on test after a late retrieval change.** It scored 0.851 on the test sample before commit `3b94efa` ("deterministic retrieval") and 0.783 after. We report the final measured number.
7. **Most runs were made from uncommitted code.** Run configs record the commit with a `-dirty` flag, e.g. `f766e47-dirty`. The flag is honest, but exact code for those runs can't be reproduced from a commit alone. In several cases the uncommitted changes were report or analysis files, not systems.
8. **The test sample has 110 no-match queries, not 100.** Our own status page said 100 for a while. The showcase now reads the number from the data.
9. **Two bugs in `report.py`, caught by the final checks.** It crashed once full-split runs sat next to 300-query runs, and it counted `s4_groups` as "our pipeline", so it could have entered the headline. Both are fixed: `s4_groups` now has its own labelled chart family, and a regression test covers mixed runs.

### Final checks
There is no GitHub Actions workflow in the repo, so we ran these locally on the merged code before submitting:
- the data loader's leak check (`python -m exactmatch.data`)
- all 8 test suites (`test_evaluate`, `test_report`, `test_errors`, `test_engine`, `test_llm`, `test_ingest`, `test_demo`, `test_normalize`)
- an import check that all 8 systems expose `find_matches`
- `compileall`

All pass.

### Link ingestion and the live demo (Harsh Kadodwala, branch `c/ingestion`, added at 15:52)
- **`ingest.py`, link to product record:** reads schema.org JSON-LD, then Open Graph/meta tags, then the page title. Fireworks is called only when the title, brand or price is still missing. `normalize.py` fills the model number. It also refuses private or local addresses.
  - Tested on a real abt.com Sony WH-CH730N page: 0.7s, no model call needed, model number `WHCH730NB`.
- **`app.py` (`python -m exactmatch.app`), the demo page:** paste a URL, see the matching listings in a **live catalog of 164 current listings** (`data/live_catalog.jsonl`).
  - For the Sony link it found no other listing of the same model, and rejected the WH-ULT900N, WH-1000XM5 and WH-1000XM6 with "model no. conflicts" evidence, at $0.0000003.
- **Things to know before demoing it:**
  - The demo uses its **own matcher**, `demo_match.py`: model-number rules first, then embeddings, then one Fireworks yes/no call per remaining candidate. It is **not** the benchmarked `s3`/`s4`, so its accuracy is not measured by the numbers above.
  - The live catalog is from **one shop (abt.com)**. Cross-shop matching shows up only in the two prepared scenes (Sony.com and Crutchfield links), and those scenes run **rules only**, with no page fetch or model call, so the recording doesn't wait.
  - **Best Buy blocks scripted requests:** the Keychron link fails with a connection reset. The demo page shows the error instead of crashing.
- **Merge fixes:** the branch was cut from the old `main`, so its two model calls used the old client (`llm.chat(model, messages, json_mode=True)` and `reply.json`). They would have crashed on the first page that needed the model; the branch's tests didn't cover that path. We fixed both call sites to the merged client and changed the default small model from Nemotron to DeepSeek V4.1 Flash, the model every reported run used. All 8 test suites pass.

## 10. Not finished

- **Using `s3`/`s4` inside the live demo** instead of the separate demo matcher, and a cross-shop live catalog.
- **`s5`, a LoRA fine-tune** of the small model on train pairs. We didn't start it.
- **Claude on the full test split, or inside our harness.** Either would make the comparison stricter.

## 11. Reproduce

```bash
pip install -r requirements.txt            # Python 3.11
cp .env.example .env                       # add FIREWORKS_API_KEY (and ANTHROPIC_API_KEY for s0_claude)
python -m exactmatch.data                  # downloads WDC, prints split stats, checks for leaks
python -m tests.test_evaluate && python -m tests.test_report && python -m tests.test_errors && python -m tests.test_engine
EM_CACHE=.cache/fresh.sqlite python -m exactmatch.evaluate --system s0 s1 s2 s3 s4 --split test --limit 300 --workers 4
python -m exactmatch.evaluate --system s3 --split test   # full split
python -m exactmatch.report                # results.md + results/*.png
python -m exactmatch.errors                # error buckets for the latest validation run
```

The default small model is DeepSeek V4.1 Flash, the model all reported runs used. `python -m exactmatch.app` starts the link demo.

## 12. Timeline (PST)

| Time | What happened |
|---|---|
| 12:00 | Repo set up. First plan: two-listing matching on Abt-Buy. |
| 12:24 | Scope changed to "link in, every match out", three roles. |
| 12:42 | `PLAN.md`: switched to WDC Products; shared schema agreed. |
| 12:54–13:02 | Data loader with leak checks, scorer, report. |
| 13:04–13:10 | Two Fireworks clients written in parallel (later merged). Llama IDs found dead; reasoning models found to return empty answers. |
| 13:21–13:44 | Claude reference built; Claude and Kimi baselines scored on the test sample. |
| 13:58 | Error analysis tool. Nemotron baseline failing on timeouts. |
| 14:37 | `s1` (hybrid retrieval). Switched to DeepSeek V4.1 Flash. |
| 14:43 | `s2` (few-shot). |
| 14:57 | `s3` (listwise + code hints) and `s4` (cascade). |
| 15:05–15:12 | Final empty-cache runs of `s0`–`s4` on the test sample. Benchmark showcase published. |
| 15:18–15:36 | Full 1,500-query test runs; `s4_groups` added and reported separately. |
| 15:37 | Code and process audit (section 9); everything merged to `main`. |
| 15:52 | Harsh's link ingestion and live demo landed; fixed for the merged client and merged to `main`. |
| 16:00 | Submission README; plan and write-up moved to `docs/`. |
