# Tuning

Practical guide to every knob the engine and pipeline expose. Where a knob
is hard-coded, the source location is given so you can fork or patch if you
need to experiment.

## Pipeline thresholds

```json
{
  "linha_fina": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4
  }
}
```

Tradeoffs:

- **Raise `conf_high`** to reduce false matches at the high tier (other
  pipeline plugins get more chances). Useful if linha-fina is competing
  against a more accurate engine downstream.
- **Lower `conf_low`** to use linha-fina as a permissive last-resort matcher
  before unparsed-input handling. Combine with a downstream confirmation
  prompt.
- The internal floor of `0.2` (in `opm.py:_calc_lf_intent`) is hard-coded.
  Anything below this is treated as "no match" regardless of tier settings.

## Training mode

```python
IntentEngine(instant_train=False)   # default
IntentEngine(instant_train=True)    # train after every register_*
```

| Mode | Use when |
|---|---|
| `False` (lazy) | Bulk-registering intents at startup. Call `train()` once at the end, or let the first inference trigger it. |
| `True` (eager) | REPL / notebook exploration where you want each registration to be immediately predictable. Costly in production. |

In the OPM pipeline, `mycroft.ready` triggers a single `train()` per
language. If your bus does not emit that event, the first user utterance
pays the training cost.

## Keyword backend

```python
KeywordFeatures(use_automatons=False)   # default: regex per entity
KeywordFeatures(use_automatons=True)    # Aho-Corasick (needs pyahocorasick)
```

Rule of thumb: switch to Aho-Corasick once a single entity has more than
~1000 keywords or the engine carries more than ~10k keywords total. The
automaton builds once and matches all patterns in a single pass.

Other knobs on `KeywordFeatures`:

- `ignore_list=[...]`: values to skip. Combined with entities whose name
  contains `_name` (for example `first_name`, `place_name`), this lets you
  ignore common-word collisions like "Mark" matching the verb "mark".
- Hard-coded **minimum match length = 3 chars** (in `keywords.py`). Shorter
  keywords are silently skipped. If you need 2-char codes (for example
  country codes), you need to fork.

## Boost / penalty weights

In `engine.py:predict`:

| Trigger | Effect |
|---|---|
| Keyword extractor matched ≥1 entity for this intent | `conf *= 1.10` |
| Intent has templates registered but none fuzzy-matched | `conf *= 0.75` |

These are hard-coded multipliers. They're deliberately asymmetric: keyword
agreement is a weak corroboration (small boost), template absence is a
stronger negative signal (larger penalty).

If you want different weighting, the cleanest approach is to subclass
`IntentEngine` and override `predict`. The components themselves don't bake
in the multipliers.

## Negative-sample cap

`DynamicBinaryClassifier.add_negative` caps the negative set at
`3 × len(positives)` and picks the most-similar negatives (highest
`token_set_ratio` to the first positive). This is the standard "hard
negative mining" pattern.

If you have intents with very few positives (≤3), this can leave the
classifier with too few negatives to learn from. Mitigations:

- Add more positive samples. Even paraphrased variants help.
- Reduce cross-skill interference by registering only the intents you need
  at a given time (the OPM pipeline supports `detach_intent` /
  `detach_skill` at runtime).

## Underlying sklearn model

`DynamicBinaryClassifier` defaults to `sklearn.neural_network.MLPClassifier`
with library defaults. You can override per-classifier:

```python
from sklearn.svm import SVC
from linha_fina.dynamic import DynamicBinaryClassifier

clf = DynamicBinaryClassifier()
clf.init_model(SVC(kernel="rbf", probability=True))
```

If you pass a model without `predict_proba` (a bare `SVC` without
`probability=True`, for instance), the classifier wraps it in
`CalibratedClassifierCV` so probabilistic output is still available.

There is **no Bayesian hyperparameter tuning**. The engine uses sklearn
defaults. For a benchmarking baseline this is intentional. If you want to
sweep `C`, kernel, layer sizes, and so on, do it externally and inject the
tuned model via `init_model`.

## Utterance length cap

`opm.py` skips utterances longer than `max_words=50`. The engine is
bag-of-words. Long inputs accumulate too many spurious token matches to
score reliably. The cap is hard-coded.

## LRU cache size

`_calc_lf_intent` is wrapped in `functools.lru_cache(maxsize=3)`. The cache
exists because OVOS calls the same utterance through `match_high`,
`match_medium`, and `match_low` in sequence. A size of 3 is enough to
absorb one utterance across all three tiers. Raise it only if you have
evidence of repeated identical utterances arriving close together.

## When tuning won't help

- The user's phrasing has *no* token overlap with any sample. The SVM sees
  an all-zero feature vector and can't pick anything. Solution: register
  more diverse samples, or accept that this query genuinely isn't covered.
- Two intents share most of their vocabulary (for example "play music" vs
  "stop music"). The bag-of-words model cannot separate them on word
  order. This is a structural limitation of linha-fina, not a tuning
  problem. Pick a matcher that uses positional features.

See [Troubleshooting](troubleshooting.md) for diagnostic steps when a
specific utterance misbehaves.

---
[← Domain pipeline](domain_pipeline.md) · [Home](index.md) · [Troubleshooting →](troubleshooting.md)
