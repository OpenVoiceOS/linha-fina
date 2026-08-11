# linha-fina

**A baseline intent matcher for benchmarking, not a production recommendation.**

`linha-fina` is essentially [padatious](https://github.com/MycroftAI/padatious)
with one-vs-rest SVMs swapping out the original neural network, layered on
top of the same template + keyword machinery. It exists as a reference point
to compare against padatious and its successors on
[`ovos-intent-benchmark`](https://github.com/OpenVoiceOS/ovos-intent-benchmark).
If you're picking an intent engine for a real assistant, use one of the
benchmarked alternatives.

```bash
pip install linha-fina
```

## Where to go next

| If you want to… | Read |
|---|---|
| Understand what intent matching *is* and why this engine is hybrid | [Concepts](concepts.md) |
| Get a working engine in 5 minutes | [Quickstart](quickstart.md) |
| Look up an API method, class, or constructor argument | [Engine API](engine.md) |
| Dig into how each component (SVM, keywords, templates) works | [Components](components.md) |
| Wire the engine into OVOS as an intent pipeline | [OPM pipeline](pipeline.md) |
| Partition intents by skill instead of one flat engine | [Domain pipeline](domain_pipeline.md) |
| Tune thresholds, training, and feature backends | [Tuning](tuning.md) |
| Debug a misfire or "why isn't this matching?" | [Troubleshooting](troubleshooting.md) |

## At a glance

```python
from linha_fina.engine import IntentEngine

eng = IntentEngine()
eng.register_intent(
    "play",
    ["play {song}", "put on {song}", "I want to hear {song}"],
    entity_samples={"song": ["africa", "hey jude", "smells like teen spirit"]},
)
eng.register_intent("stop", ["stop", "shut up", "be quiet"])
eng.register_intent("next", ["next track", "skip", "next song"])

match = eng.calc_intent("play africa")
# match.name == "play"
# match.slots == {"song": "africa"}
# match.conf  ~ 0.9
```

A single call to `calc_intent` runs three layers internally: an SVM over
keyword features picks candidate intents, a fuzzy template matcher refines
their confidence and extracts slots, and a keyword extractor fills in any
slots the template missed. See [Concepts](concepts.md) for the full picture.
