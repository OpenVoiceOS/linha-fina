# Quickstart

A complete walkthrough: install, train, predict, extract slots, and integrate
with OVOS. By the end you'll have a working three-intent engine and know
where to go to deepen each part.

## 1. Install

```bash
pip install linha-fina
```

Optional extras:

```bash
pip install linha-fina[test]        # pytest + pytest-cov for running the test suite
pip install pyahocorasick           # faster keyword extraction (see Tuning)
```

## 2. Your first engine

```python
from linha_fina.engine import IntentEngine

eng = IntentEngine()

eng.register_intent(
    "greet",
    ["hello", "hi there", "hey", "good morning"],
)
eng.register_intent(
    "bye",
    ["goodbye", "see you", "bye", "talk to you later"],
)
eng.register_intent(
    "thanks",
    ["thank you", "thanks", "much appreciated"],
)

eng.train()

print(eng.calc_intent("hi"))
# IntentMatch(name='greet', conf=0.87, slots={})

print(eng.calc_intent("see ya"))
# IntentMatch(name='bye',   conf=0.71, slots={})
```

Two things to notice:

- You **need ≥3 intents** registered before `train()` will fit a useful SVM.
  Below that, the engine falls back to template-only matching.
- `train()` is optional — if you skip it, the first call to `calc_intent`
  trains automatically (lazy training). Calling it explicitly during startup
  trades a small upfront cost for a snappier first response.

## 3. Adding slots

Slots are placeholders in your samples written as `{name}`:

```python
eng = IntentEngine()

eng.register_intent(
    "play",
    ["play {song}", "put on {song}", "I want to hear {song}"],
    entity_samples={"song": ["africa", "hey jude", "smells like teen spirit"]},
)
eng.register_intent("stop", ["stop", "pause", "shut up"])
eng.register_intent("next", ["next track", "skip", "next song"])

eng.train()

m = eng.calc_intent("put on africa")
# m.name  == "play"
# m.slots == {"song": "africa"}
# m.conf  ~ 0.9
```

`entity_samples` does double duty:

1. It seeds the **template matcher**, so `"play {song}"` can fuzzy-match the
   actual utterance and extract the slot span.
2. It seeds the **keyword extractor**, so even if the template misses (user
   said something unexpected) the slot can still be filled by literal keyword
   match.

You can register entities separately too:

```python
eng.register_entity("song", ["wonderwall", "yesterday"])
```

This appends to the global keyword vocabulary; pass `intent_name="play"` to
scope the entity to one intent.

## 4. Inspecting predictions

`predict` returns the top-N candidates with confidences, useful for debugging:

```python
for m in eng.predict("put on africa", top_n=3):
    print(f"{m.name:10s} conf={m.conf:.2f}  slots={m.slots}")
# play       conf=0.90  slots={'song': 'africa'}
# next       conf=0.04  slots={}
# stop       conf=0.02  slots={}
```

If "play" doesn't win, see [Troubleshooting](troubleshooting.md).

## 5. Removing and rebuilding

Intents and entities can be removed at runtime:

```python
eng.remove_intent("greet")
eng.remove_entity("song", intent_name="play")
```

The next `train()` (or first inference) rebuilds affected classifiers only.

## 6. Wiring into OVOS

The OPM plugin handles all of the above behind the messagebus. Enable it in
`mycroft.conf`:

```json
{
  "intents": {
    "pipeline": ["ovos-linha-fina-pipeline-plugin"]
  },
  "linha_fina": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4
  }
}
```

Skills register intents using the standard padatious messagebus protocol;
the plugin transparently routes them into a per-language `IntentEngine`. See
[OPM pipeline](pipeline.md) for the full message contract and confidence
tiers.

## 7. Where to go next

- **API reference:** [Engine API](engine.md) lists every public method and
  argument.
- **Internals:** [Components](components.md) goes one level deeper into the
  SVM, template, and keyword subsystems — useful when tuning or extending.
- **Tuning:** [Tuning](tuning.md) covers thresholds, training modes, and the
  Aho-Corasick keyword backend.

---
[← Concepts](concepts.md) · [Home](index.md) · [Engine API →](engine.md)
