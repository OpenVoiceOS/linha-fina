# Components

`IntentEngine` is the public face. Under the hood it composes three
independently usable pieces. This page documents each one so you can use
them standalone, replace them, or understand why a prediction came out the
way it did.

| Module | Class | Role |
|---|---|---|
| `linha_fina.keywords` | `KeywordFeatures` | Entity keyword vocabulary + extraction + one-hot featurization |
| `linha_fina.templates` | `TemplateMatcher` + expansion helpers | Slot-filled pattern matching with fuzzy scoring |
| `linha_fina.dynamic` | `DynamicClassifier`, `DynamicBinaryClassifier` | One-vs-rest SVM stack with auto-managed negatives |

## KeywordFeatures

A vocabulary of `{entity_name: [literal strings]}` plus utilities to:

- extract entities from a sentence (returns `{entity: longest_match}`)
- produce a binary one-hot vector over the known vocabulary (the feature
  representation the SVM consumes)
- persist/restore via joblib

### Construction

```python
from linha_fina.keywords import KeywordFeatures

kw = KeywordFeatures(
    csv_path=None,             # bulk-load "entity,keyword" rows
    ignore_list=None,          # words to skip (esp. common names)
    use_automatons=None,       # None = regex; True = Aho-Corasick (needs pyahocorasick)
)
```

### Registering & extracting

```python
kw.register_entity("color", ["red", "green", "blue"])
kw.register_entity("song",  ["africa", "hey jude"])

kw.extract("play africa now")
# {'song': 'africa'}

list(kw.match("turn the lights red"))
# [('color', 'red')]
```

`extract` returns the longest match per entity (so `"toto"` won't shadow
`"africa by toto"` if both are registered). `match` yields every match.

### One-hot featurization

```python
kw.one_hot_encode("play africa")
# [0, 1, 0, ...]   # binary presence vector over the vocabulary
```

This is what the SVM trains and predicts on. It is order-agnostic by
construction. That is the major limitation flowing up to the engine.

### Backends

- **Regex (`use_automatons=False`):** default. Builds one regex per entity.
  Fine for hundreds of keywords, gets slow in the thousands.
- **Aho-Corasick (`use_automatons=True`):** requires `pip install
  pyahocorasick`. Builds a single automaton matching all patterns in one pass
  over the input. Recommended once your vocabulary exceeds ~1k entries.

Both backends skip matches shorter than 3 characters and (when the entity
name contains `_name`) skip values that appear in `ignore_list`.

## Templates

A template is a sample utterance with `{slot}` placeholders, optionally with
expansion syntax for variants:

```
"[hello,] (call me|my name is) {name}"
```

Expands to:

```
"call me {name}"
"my name is {name}"
"hello, call me {name}"
"hello, my name is {name}"
```

### TemplateMatcher

```python
from linha_fina.templates import TemplateMatcher

tm = TemplateMatcher()
tm.add_templates([
    "play {song}",
    "put on {song}",
    "I want to hear {song}",
])

tm.match("put on africa")
# [{'song': 'africa'}]
```

`match` returns extracted slot dicts sorted by descending fuzzy score
(`rapidfuzz.fuzz.token_set_ratio`). The first hit is the best match.

### Expansion helpers

```python
from linha_fina.templates import expand_template, expand_slots

expand_template("[hi,] (call me|I'm) {name}")
# ['call me {name}', "I'm {name}", 'hi, call me {name}', "hi, I'm {name}"]

expand_slots("play {song}", {"song": ["africa", "hey jude"]})
# ['play africa', 'play hey jude']
```

`expand_slots` is what the engine uses internally to turn `entity_samples`
into additional positive training samples for the SVM.

### Mechanics

Matching combines two libraries:

- **simplematch** decides whether a template *can* fit (glob-like
  pattern + slot capture).
- **rapidfuzz** ranks matched templates by token-set similarity to the input.

Templates that don't fit at all are dropped. The rest are scored and sorted.

## DynamicClassifier

A one-vs-rest stack of binary SVMs. Each registered label gets its own
`DynamicBinaryClassifier`. Predicting an utterance evaluates all of them
and returns `{label: probability}`.

### DynamicBinaryClassifier

```python
from linha_fina.dynamic import DynamicBinaryClassifier

clf = DynamicBinaryClassifier()
clf.add_positive(["play africa", "put on hey jude", "play wonderwall"])
clf.add_negative(["what's the weather", "set a timer", "turn off the lights"])
clf.train()
clf.predict("play yesterday")
# 0.87
```

Internals:

- Default model is `sklearn.neural_network.MLPClassifier`. If you pass a
  bare `SVC` via `init_model`, the classifier wraps it in
  `CalibratedClassifierCV` so `predict_proba` is available.
- `add_negative` caps the negative set at `3 × len(positives)` and selects
  the most ambiguous ones (highest `token_set_ratio` to the first positive),
  on the theory that the *hardest* negatives produce the tightest decision
  boundary.
- Featurization is one-hot via a shared `KeywordFeatures` instance, the
  same vocabulary used everywhere in the engine.

### DynamicClassifier

```python
from linha_fina.dynamic import DynamicClassifier

clf = DynamicClassifier(instant_train=False)
clf.add_label("play", ["play africa", "put on hey jude"])
clf.add_label("stop", ["stop", "shut up", "pause"])
clf.add_label("next", ["next track", "skip"])
clf.train()

clf.predict("play wonderwall")
# {'play': 0.84, 'stop': 0.03, 'next': 0.11}
```

- `train()` runs the per-label fits in a `ThreadPoolExecutor` (so a 50-intent
  engine trains roughly as fast as the slowest single classifier, not 50×
  one).
- `eval_fp()` evaluates the false-positive rate by predicting every
  unregistered sample's classifier on every other label. Use it when tuning
  thresholds.
- Requires **≥3 labels** before `train()` will produce useful classifiers
  (the negatives need somewhere to come from).

## Combining them

`IntentEngine.predict` does the orchestration:

1. Featurize the utterance once via `KeywordFeatures.one_hot_encode`.
2. Get per-intent probabilities from `DynamicClassifier.predict`.
3. For each candidate intent, run `TemplateMatcher.match` (boost or penalty)
   and `KeywordFeatures.extract` (slot extraction, small boost on hit).
4. Sort, return.

If you need different orchestration, for example template-first with the
SVM as tiebreak, or different boost and penalty weights, subclass
`IntentEngine` and override `predict`. The underlying components have
no opinion about how they are combined.

---
[← Engine API](engine.md) · [Home](index.md) · [OPM pipeline →](pipeline.md)
