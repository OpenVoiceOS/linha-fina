# Engine API

Reference for `linha_fina.engine.IntentEngine`, the high-level orchestrator
that combines the SVM classifier, template matcher, and keyword extractor.

For deeper internals see [Components](components.md). For the OVOS plugin
wrapper see [OPM pipeline](pipeline.md).

## Import

```python
from linha_fina.engine import IntentEngine, IntentMatch
```

## `IntentEngine(instant_train=False)`

Create a new engine.

| arg | type | default | meaning |
|---|---|---|---|
| `instant_train` | bool | `False` | If `True`, call `train()` automatically after every `register_intent` / `register_entity`. Convenient for REPL exploration. Expensive in production, where you want batched registration followed by a single `train()`. |

When `instant_train=False` (the default), training is **lazy**: the engine
sets an internal "needs training" flag on each registration and only fits
when the next `calc_intent` / `predict` call is made.

## Registering intents

### `register_intent(name, samples, entity_samples=None)`

```python
eng.register_intent(
    "play",
    samples=["play {song}", "put on {song}"],
    entity_samples={"song": ["africa", "hey jude"]},
)
```

| arg | type | meaning |
|---|---|---|
| `name` | `str` | Intent label. In the OPM pipeline this is `"{skill_id}:{intent_name}"`. In raw use, any string works. |
| `samples` | `list[str]` | Example utterances. May contain `{slot}` placeholders. Templates with `[optional]` or `(alt\|alt)` syntax are auto-expanded into all permutations before being fed to the matcher. |
| `entity_samples` | `dict[str, list[str]] \| None` | Per-slot value lists used both as keyword vocabulary and as training data for the SVM (slots are filled with each value to create more positive samples). |

The samples are expanded by `templates.expand_slots` before training, so
`"play {song}"` with `entity_samples={"song": ["africa", "hey jude"]}` yields
the training corpus `["play africa", "play hey jude"]` in addition to the
template `"play {song}"` itself.

### `remove_intent(name)`

Removes the intent and its associated template matcher, keyword features,
and per-intent binary classifier. The engine is marked as needing retraining.

## Registering entities

### `register_entity(name, samples, intent_name=None)`

Add keyword values for an entity slot.

| arg | meaning |
|---|---|
| `name` | Entity label, e.g. `"song"`, `"color"`. |
| `samples` | List of literal strings to match. |
| `intent_name` | If given, scope the entity to that intent only. If `None`, the entity is global and may be matched for any intent that declares it as a slot. |

Use this when entity values arrive separately from intent samples. For
example, a skill registers `"play"` at startup and pushes new songs into
the `song` entity as the user's library is indexed.

### `remove_entity(name, intent_name=None)`

Inverse of `register_entity`.

## Training

### `train()`

Fits the SVM(s) over the current intent and entity state. Idempotent and
safe to call repeatedly. The engine tracks dirty flags and skips work if
nothing changed.

Raises if fewer than 3 intents are registered (multi-class one-vs-rest
requires at least three positive classes to be meaningful).

## Inference

### `calc_intent(query) -> IntentMatch | None`

Top-1 prediction. Returns an `IntentMatch` or `None` if no intent crossed
the engine's internal floor (~0.2).

### `predict(query, top_n=3) -> list[IntentMatch]`

Top-N predictions, sorted by descending confidence. Useful for debugging,
re-ranking, or pipelines that want to consider multiple candidates.

```python
@dataclass
class IntentMatch:
    name: str                       # the intent label
    slots: dict[str, str]           # extracted slot values, may be empty
    conf: float                     # 0.0–1.0 after all boost/penalty steps
```

Confidence is computed as:

```
conf = svm_probability
       * 1.10  if keyword extractor matched any entity for this intent
       * 0.75  if the intent has templates but none of them fuzzy-matched
```

Both adjustments are multiplicative and may stack. The +10% / −25% values
are deliberate: keyword agreement is a weak corroboration (small boost),
template absence is a stronger negative signal (larger penalty).

## Full example

```python
from linha_fina.engine import IntentEngine

eng = IntentEngine(instant_train=False)

eng.register_intent(
    "play",
    ["play {song}", "put on {song}", "I want to hear {song}"],
    entity_samples={"song": ["africa", "hey jude", "wonderwall"]},
)
eng.register_intent("stop", ["stop", "shut up", "pause"])
eng.register_intent("next", ["next track", "skip", "next song"])

eng.train()

for utt in ["put on africa", "shut up", "skip this one"]:
    m = eng.calc_intent(utt)
    print(f"{utt!r:30s} → {m.name:6s} conf={m.conf:.2f} slots={m.slots}")
```

```
'put on africa'                → play   conf=0.91 slots={'song': 'africa'}
'shut up'                      → stop   conf=0.84 slots={}
'skip this one'                → next   conf=0.72 slots={}
```

---
[← Quickstart](quickstart.md) · [Home](index.md) · [Components →](components.md)
