# ExactMatch

**Catching duplicate product listings with a small open model, more accurately and cheaply than an off-the-shelf LLM.**

Marketplaces, price-comparison sites and retail catalog teams keep getting the same product listed under different titles, descriptions and prices by different sellers. *"Sony Bravia 46in LCD KDL-46V5100"* and *"Sony 46" BRAVIA V-Series Full HD LCD TV - KDL46V5100"* are the same SKU. Missed duplicates cause:

- **Fragmented listings**: reviews, inventory and buy-box get split across copies, which hurts conversion.
- **Wrong price comparisons**: price-comparison sites show the wrong "best price".
- **Manual review cost**: catalog ops teams pay people to check candidate pairs by hand.

ExactMatch takes two product records and decides whether they are the **same real-world product**. It wraps a small open model (served on Fireworks AI) in a harness that adds normalization, retrieved examples, consistency checks and routing to a larger model. We measure every version on the same held-out public benchmark.

---

## How we map to the judging criteria

| Weight | Criterion | How we cover it |
|---|---|---|
| 40% | Benchmark evaluation | Fixed public test split (Abt-Buy, plus Amazon-Google if time allows), with hundreds to thousands of labeled pairs. Primary metric is **F1 on the "match" class**. Every system runs on the same pairs with the same scorer. We report baseline, each improvement, and the gain. |
| 25% | Business value | Clear user (marketplace / catalog ops). We report **F1, cost per 1k pairs and latency** for each system, and translate the error reduction into reviewer-hours saved. |
| 20% | Technical execution | Harness of normalization, retrieved few-shot examples, rule checks, a confidence-based cascade, and an optional LoRA fine-tune trained only on the train split. |
| 10% | Presentation & demo | Live demo: paste two listings and get a verdict with the evidence (e.g. model numbers that match or conflict), plus a before/after results table. |
| 5% | Documentation | One command reproduces each row of the results table. Every run saves its config, prompt, outputs and metrics. |

---

## Data

| Dataset | What it is | Why |
|---|---|---|
| **Abt-Buy** (primary) | Electronics listings from two retailers. Mostly free text, and model numbers are buried in it. | Small LLMs do poorly here out of the box, so there is clear headroom. |
| **Amazon-Google** (secondary) | Software products, with noisy titles and many near-duplicates (versions and editions). | Second dataset to show the gain generalizes. |

We use the standard **train / valid / test splits from the DeepMatcher benchmark** (Magellan / UW-Madison data repository). Leipzig's entity-resolution benchmark page hosts the same source data. If one source is unreachable, use the other.

**Rules (non-negotiable):**
1. **The test split is frozen.** It is used only for final scoring. Nobody looks at test errors to tweak prompts.
2. Prompt iteration, threshold tuning and the choice of few-shot examples use **train/valid only**.
3. The few-shot retrieval index and the fine-tuning data come from **train only**.
4. Report the exact number of test pairs and the positive rate next to every score.

---

## Systems (the ladder we climb)

Each rung is a separate, runnable system ID. Each one is scored on the same test set, so the gain from each idea is visible.

| ID | System | Idea |
|---|---|---|
| `s0` | **Baseline** | Small open model, zero-shot: *"Are these the same product? yes/no"*. This is "off the shelf". |
| `s0-big` | **Reference** | Large open model, zero-shot. Shows whether our small-model harness matches or beats a model that costs much more. |
| `s1` | **Structured prompt + normalization** | Pull out brand, model/part number, capacity/size and color with regex and the LLM. Normalize them (case, hyphens, units). Prompt with a short matching rubric and force JSON output (`{"match": bool, "confidence": 0-1, "evidence": ...}`). |
| `s2` | **+ Retrieved few-shot** | Embed the pair, then retrieve the k most similar *labeled train pairs* (both matches and hard non-matches) and include them as examples. |
| `s3` | **+ Rule checks** | Hard signals override the LLM: model numbers that conflict mean **no match**. Exact normalized model numbers with the same brand mean **match**. The LLM decides only the ambiguous middle. Optionally take a majority vote over 3 samples. |
| `s4` | **+ Cascade / routing** | Pairs where the small model has low confidence go to the large model. Report the share escalated, F1 and the blended cost. |
| `s5` *(optional)* | **+ LoRA fine-tune** | Fine-tune the small model on train pairs (Fireworks fine-tuning) and use it inside the `s3`/`s4` harness. |

**Expected story:** `s0` is weak on Abt-Buy because model numbers are hidden in long descriptions. `s1`–`s3` should give the biggest jump. `s4` gives most of the large model's quality at a fraction of its cost.

---

## Evaluation protocol

- **Primary metric:** F1 on the positive (match) class. Precision, recall and accuracy are reported as well.
- **Ops metrics:** cost per 1k pairs (from Fireworks token usage × price), mean and p95 latency, and the escalation rate for `s4`.
- **Reliability:** report run-to-run variation (mean ± std over 3 seeds or samples) for the final systems, if budget allows. Use temperature 0 where possible.
- **Error analysis:** done on the **valid** split only. Bucket errors as: model-number conflict, variant/edition, bundle vs. single item, accessory vs. main product, missing information.
- **Business translation:** *false negatives* mean duplicates stay live (fragmented listings). *False positives* mean distinct products get wrongly merged (more expensive, because it shows customers the wrong item). Report both, plus "human review pairs avoided per 10k candidates".

### Reproducibility
Every run writes to `runs/<timestamp>_<system>_<split>/`:
```
config.json        # system id, model ids, temperature, k, thresholds, git commit
prompt.txt         # exact prompt template(s)
predictions.jsonl  # pair id, label, prediction, confidence, evidence, tokens, latency
metrics.json       # P / R / F1 / acc / cost / latency
```
`results.md` is generated from `runs/` and is never edited by hand.

---

## Repo layout (planned)

```
exactmatch/
  data.py         # download + load splits -> list[Pair]
  normalize.py    # brand / model-number / unit extraction + normalization
  llm.py          # Fireworks client: retries, JSON parsing, token + cost accounting, caching
  retrieval.py    # embedding index over train pairs, kNN few-shot selection
  systems/        # s0.py, s1.py, ... each exposes predict(pairs) -> list[Prediction]
  evaluate.py     # scorer (single source of truth) + run logging
  report.py       # builds results.md from runs/
  demo.py         # small UI / CLI: two listings in, verdict + evidence out
data/             # downloaded datasets (git-ignored)
runs/             # run outputs (commit the final ones)
```

**Shared interface:** every system implements

```python
def predict(pairs: list[Pair]) -> list[Prediction]  # Prediction(match: bool, confidence: float, evidence: str, cost_usd: float, latency_s: float)
```

and is run with

```bash
python -m exactmatch.evaluate --system s2 --dataset abt-buy --split test
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FIREWORKS_API_KEY=...   # never commit this
python -m exactmatch.data --download
```

**Budget:** $50 of Fireworks credits. A full test pass with a small model costs well under $1, and a large model costs a few dollars. Cache every LLM call (key = model + prompt hash) so reruns are free. Check spend before any large-model or fine-tuning job.

---

## Timeline (submission deadline 4:30 pm)

| By | Milestone |
|---|---|
| 12:00 | Data loaded, scorer working, `s0` baseline scored on **test**. This is the critical path. |
| 1:00 | `s1` (normalization + structured prompt) scored |
| 2:00 | `s2` (retrieval few-shot) + `s3` (rule checks) scored |
| 3:00 | `s4` cascade scored; `s5` fine-tune only if it is already running |
| **3:30** | **Feature freeze.** Final runs with 3 seeds, generate `results.md` |
| 4:15 | Demo, slides and README results section done; submit |

---

## Working in parallel (2 people)

The split follows the code boundaries, so the two of you rarely edit the same file.

| | **Person A: data, eval and rules** | **Person B: model and harness** |
|---|---|---|
| First 45 min | `data.py`, `evaluate.py` (scorer + run logging), `requirements.txt`, `.gitignore` | `llm.py` (Fireworks client, JSON parsing, cost accounting, cache); pick the small and large model IDs |
| Then | `s0` + `s0-big` baseline runs; `normalize.py` + `s1` | `retrieval.py` + `s2` (retrieved few-shot) |
| Then | `s3` rule checks; error analysis on **valid** | `s4` cascade; kick off `s5` fine-tune early if attempting it |
| Last hour | `report.py` → `results.md`, README results section | `demo.py`, slides / demo script |

**Ground rules**
- **Agree on `Pair` / `Prediction` and the `predict()` signature in the first 10 minutes.** B can then build against a stub data loader before A's is done.
- Branch per person (`a/...`, `b/...`), small PRs merged into `main` often. Pull before starting each new system.
- **Only Person A edits `evaluate.py` once the baseline is scored.** Changing the scorer mid-day invalidates the comparison.
- Neither person looks at **test** errors. Tune on valid.
- Commit final `runs/` directories so every number in `results.md` can be traced back to its run.
