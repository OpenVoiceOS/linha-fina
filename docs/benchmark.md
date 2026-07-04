# Benchmark

`linha-fina` ships a comparative accuracy benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports linha-fina — in all three variants — alongside a fixed set of external baselines, so results are directly comparable across the OVOS intent-engine family.

---

## Headline results — `intents-for-eval`

50 intents, 1700 labelled test cases, 50 off-topic (`far_ood`) cases.

| Engine | def F_0.5 | **opt F_0.5** | opt thr | opt FP | **Rec @ P≥99%** |
|---|---|---|---|---|---|
| padatious (neural) | 0.903 | 0.928 | 0.18 | 8 | **73.4%** |
| **linha-fina flat** | 0.916 | **0.917** | 0.47 | 17 | **70.0%** |
| linha-fina domain (parallel) | 0.914 | 0.915 | 0.52 | **12** | 70.4% |
| nebulento `damerau` | 0.909 | 0.918 | 0.43 | 26 | 62.0% |
| linha-fina hierarchical (two-stage) | 0.893 | 0.895 | 0.44 | 7 | 64.4% |
| padaos (regex) | 0.832 | 0.832 | 0.50 | 1 | 0.0% (recall ≤ 50%) |

**linha-fina flat reaches F_0.5 = 0.917 — tied with the best fuzzy engine (nebulento `damerau` at 0.918) and within 0.011 of padatious's neural baseline.** Its R@P≥99% of 70.0% trails only padatious in the non-trained family.

Read the per-section deep-dive below for what those numbers mean, when to pick which variant, and why the per-skill SVM gives linha-fina a small but real edge over pure fuzzy string matchers.

---

## Why F_0.5 and not F1

A voice assistant's two failure modes are not symmetric:

- **False positive** — the wrong intent fires, the skill executes the wrong action, the assistant says the wrong thing. There is no recovery layer above the intent service that can catch this; the user has to notice, abort, and re-ask.
- **False negative** — no intent fires. OVOS hands the utterance to its fallback chain: common-query, the LLM fallback, online search. These exist precisely to handle "I don't know what you meant."

The cost ratio is roughly 5–10× in favour of false negatives. F1 (which weights precision and recall equally) is the wrong summary metric. **F_β with β=0.5** weights precision twice as recall and is the right summary for OVOS.

We also report **Rec@P≥99%** — the recall achievable once the threshold is tuned to keep precision at or above 99%. This is the operating point a production OVOS install actually picks: "give me the most coverage you can while letting through at most 1% wrong matches."

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances).

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits including a 50-row `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of MASSIVE; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | Tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

### Slot handling

Every `{slot}` placeholder in the templates ships with a list of example values. linha-fina's `IntentEngine.register_intent` accepts `entity_samples={slot: [values, ...]}` directly — slot values are passed in at registration time and `TemplateMatcher` extracts them at inference. No external slot-fill helper is needed for linha-fina.

---

## Engines

Three external engines are **fixed baselines** — the same engines and settings used in every OVOS intent-engine benchmark. Three linha-fina rows are the **subject**.

| Engine | Role | Notes |
|---|---|---|
| `padaos` | baseline | regex-based exact matcher |
| `padatious` | baseline | neural matcher (requires `train()` pass) |
| `nebulento` | baseline | fuzzy string matcher, `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| `linha-fina flat` | subject | one SVM per skill, global argmax |
| `linha-fina domain` (parallel) | subject | per-skill SVMs grouped by domain; scored in parallel, global argmax |
| `linha-fina hierarchical` (two-stage) | subject | top-level domain classifier routes to one per-domain SVM ensemble |

linha-fina trains a small per-skill SVM during `register_intent` (no `train()` step is exposed — training is incremental). The baselines below take milliseconds; linha-fina takes seconds to register the full corpus and ~60 ms per query to run all the SVMs.

---

## Deep-dive: where linha-fina wins and where it doesn't

### vs nebulento (the closest competitor)

Both linha-fina and nebulento are non-neural, non-trained-per-domain matchers. They score essentially the same on F_0.5 (0.917 vs 0.918) but with different precision/recall profiles:

| | linha-fina flat | nebulento `damerau` |
|---|---|---|
| F_0.5 (opt) | 0.917 | 0.918 |
| FP (opt) | 17 | 26 |
| R@P≥99% | **70.0%** | 62.0% |
| latency (median) | 58 ms | 3 ms |

linha-fina's per-skill SVM gives it a real precision advantage — **at the strict 99% precision floor, it covers 8pp more queries than nebulento** (70% vs 62%). The SVM's decision boundary discriminates between in-distribution and out-of-distribution embeddings in a way that pure string similarity can't. The cost is 20× the latency (60 ms vs 3 ms per query).

### vs padatious (the neural baseline)

padatious narrowly wins on F_0.5 (0.928 vs 0.917) and R@P≥99% (73.4% vs 70.0%). That's roughly a 3pp coverage gap at the production operating point. linha-fina trades it for two operational advantages:

- **No training pass.** Padatious requires a one-shot `train()` call (~13 s on this corpus); linha-fina is incrementally trained as intents register.
- **Per-skill modularity.** Adding or removing a skill in linha-fina retrains exactly one SVM. Padatious flat retrains the whole model.

### Domain (parallel) — does the per-skill grouping help?

The Domain variant groups intents by their `<domain>:<intent>` prefix and scores per-domain ensembles in parallel before taking a global argmax. The numbers are nearly identical to flat:

| | flat | Domain (parallel) |
|---|---|---|
| F_0.5 (opt) | 0.917 | 0.915 |
| FP (opt) | 17 | **12** |
| R@P≥99% | 70.0% | 70.4% |

Domain trades 0.002 F_0.5 for **5 fewer false positives** — the per-domain context makes each SVM slightly more conservative outside its domain. For OVOS, where FPs cost more than coverage, this is a win: pick Domain over flat when you want one of: (a) tighter FP control, (b) per-skill retrain without touching unrelated SVMs.

### Hierarchical — the precision tool

The Hierarchical variant runs a top-level domain classifier first and then resolves intent inside the chosen domain only:

| | flat | Hierarchical |
|---|---|---|
| F_0.5 (opt) | 0.917 | 0.895 |
| FP (opt) | 17 | **7** |
| R@P≥99% | 70.0% | 64.4% |
| latency | 58 ms | 42 ms |

Hierarchical pays ~2pp F_0.5 for **less than half the FPs of flat** and lower latency (only one per-domain ensemble runs per query). It is the right choice when off-topic rejection is paramount and a small recall drop is acceptable — e.g., for an OVOS install with aggressive LLM fallback that wants the intent layer to fire only on high-confidence matches.

### Threshold calibration — already well-tuned

| Variant | calibration delta on F_0.5 |
|---|---|
| linha-fina flat | +0.001 |
| linha-fina domain | +0.001 |
| linha-fina hierarchical | +0.002 |

The shipped default `threshold=0.5` is essentially F_0.5-optimal for every linha-fina variant. SVMs produce well-calibrated confidence scores by construction (their `decision_function` distance from the hyperplane maps cleanly to a probability via the sigmoid), so the engine doesn't benefit much from per-deployment threshold tuning.

This contrasts with markov (calibration moves F_0.5 by +0.24) and m2v prototype (`+0.06`), where calibration is essential.

---

## flat vs Domain vs Hierarchical — when to pick which

| Property | flat | Domain (parallel) | Hierarchical (two-stage) |
|---|---|---|---|
| F_0.5 (opt) | **0.917** | 0.915 | 0.895 |
| FP (opt) | 17 | 12 | **7** |
| R@P≥99% | 70.0% | **70.4%** | 64.4% |
| latency (median) | 58 ms | 58 ms | **42 ms** |
| Add/remove a skill | one SVM retrain | one SVM retrain | one SVM retrain + router retrain |
| Recommended when | default | tighter FP control | aggressive FP rejection + fallback chain |

---

## Reproducing

```bash
pip install linha-fina[benchmark]
python benchmark/compare.py intents-for-eval   # ~3 minutes
python benchmark/compare.py massive            # ~12 minutes
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards). linha-fina's per-skill SVMs are trained inline during registration; baseline engines train on first query (padaos/nebulento) or via a single `train()` call (padatious).

## How metrics are calculated

Source: `compute_metrics`, `calibrate_threshold`, `fbeta`, `recall_at_precision` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2·P·R / (P + R)
- **F_0.5** = 1.25·P·R / (0.25·P + R) — weights precision 2× recall (default summary metric for OVOS)
- **Rec@P≥99%** = max recall achievable by sweeping the threshold while keeping precision ≥ 99%
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf ≥ threshold`. A no-match case is correct only when the engine returns `None` or a confidence below threshold.
