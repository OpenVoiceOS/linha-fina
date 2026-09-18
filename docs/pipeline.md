# OPM pipeline

`LinhaFinaPipeline` plugs `IntentEngine` into the OpenVoiceOS intent service
via the [OPM](https://github.com/OpenVoiceOS/OVOS-plugin-manager) plugin
interface. It listens on the OVOS messagebus, maintains one engine per
language, and exposes the three confidence-tier matchers OVOS expects.

> **Reminder:** linha-fina is a benchmarking baseline. The pipeline plugin
> ships so it can be slotted into a real OVOS install for A/B comparison
> against padatious and successors, not because it is the recommended
> default. See [Concepts](concepts.md) for context.

## Entry point

```
opm.pipeline:
  ovos-linha-fina-pipeline-plugin = linha_fina.opm:LinhaFinaPipeline
```

## Configuration

In `mycroft.conf`:

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

| key | default | meaning |
|---|---|---|
| `conf_high` | `0.8` | Threshold for `match_high`. The pipeline returns a match only if intent confidence exceeds this. |
| `conf_med` | `0.6` | Threshold for `match_medium`. |
| `conf_low` | `0.4` | Threshold for `match_low`. |

In addition, the pipeline reads `lang` (primary language) and
`secondary_langs` from the top-level OVOS config. Each language gets its own
`IntentEngine` instance, populated independently as skills register intents
for that language.

## Confidence tiers

OVOS's `ConfidenceMatcherPipeline` contract has three matchers. Each tier
runs in a different phase of the intent service's pipeline, so high-
confidence engines short-circuit before low-confidence ones get a chance.

| Method | Phase | Use |
|---|---|---|
| `match_high(utterances, lang, message)` | Tried first, before any low-confidence engine | Only fires on the most certain matches |
| `match_medium(utterances, lang, message)` | Mid-pipeline | Catches utterances the high tier passed on |
| `match_low(utterances, lang, message)` | Last-resort | Permissive, useful as a fallback before unparsed |

All three call into the same `_match_level` helper with a different
threshold. Below `conf_high` the high matcher returns `None` and OVOS
proceeds. Below all three, the engine gives up entirely and the next
pipeline plugin gets a turn.

There is also a hard floor of `0.2` applied inside `_calc_lf_intent`.
Predictions below this never escape the pipeline, regardless of the tier
thresholds.

## Messagebus protocol

The pipeline reuses padatious's bus contract, so existing skills work
without changes:

| Event | Payload | Effect |
|---|---|---|
| `padatious:register_intent` | `{name, samples, entity_samples?, lang?}` | Calls `IntentEngine.register_intent` on the matching language container. |
| `padatious:register_entity` | `{name, samples, lang?}` | Calls `IntentEngine.register_entity`. |
| `detach_intent` | `{intent_name}` | Removes the intent from all language containers. |
| `detach_skill` | `{skill_id}` | Removes every intent whose name starts with `{skill_id}:`. |
| `mycroft.ready` | none | Triggers `train()` on every language container. Without this, training is lazy and the first utterance pays the cost. |

Intent names follow the convention `"{skill_id}:{intent_name}"`. The
pipeline uses the prefix to populate `IntentHandlerMatch.skill_id`.

## Language routing

Each inference call carries a `lang` argument. The pipeline:

1. Checks for an exact-language container.
2. Falls back to `langcodes.closest_match(lang, available, max_distance=10)`
   if no exact match exists. This handles cases like
   `en-AU` → `en-US` gracefully.
3. Returns `None` if no container is within the distance threshold.

## Session-aware filtering

`_calc_lf_intent` consults `SessionManager.get(message)` to honour the
session's `blacklisted_intents` and `blacklisted_skills`. Matches whose
intent or skill prefix is blacklisted in the active session are filtered
out *before* threshold checks, so a low-confidence allowed match can win
over a high-confidence blacklisted one.

## Caching

`_calc_lf_intent` is wrapped in `functools.lru_cache(maxsize=3)`. Repeated
inference on the same utterance (common with multi-tier pipelines where the
same string is shown to high/medium/low matchers in turn) avoids redundant
SVM evaluation.

## Limits

- `max_words=50`: utterances longer than 50 whitespace-separated tokens are
  skipped without inference. linha-fina is bag-of-words and would produce
  noisy scores on long inputs.
- One engine per language. If you have 50 intents across 10 skills in 3
  languages, you hold 3 engines × 50 intents × ~1 small SVM each = 150
  binary classifiers. They're cheap individually.

## Going further

- [Tuning](tuning.md): a practical knob-by-knob guide.
- [Troubleshooting](troubleshooting.md): "why didn't my intent fire?"

---
[← Components](components.md) · [Home](index.md) · [Domain pipeline →](domain_pipeline.md)
