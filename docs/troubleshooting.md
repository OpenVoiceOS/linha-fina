# Troubleshooting

A diagnostic flowchart for "this utterance isn't matching what I expect."
Each section starts with the symptom and ends with the most likely fix.

## My intent isn't firing at all

`calc_intent` returns `None`, or your handler never runs.

**Step 1: dump the top candidates.**

```python
for m in eng.predict(utterance, top_n=5):
    print(f"{m.name:20s} conf={m.conf:.3f}  slots={m.slots}")
```

If your intent doesn't appear at all, the SVM scored it below the engine's
internal `0.2` floor. If it appears but with low confidence, see the next
section.

**Step 2: check intent count.** The SVM needs ≥3 intents to train. With 1
or 2 you'll get template-only matching, which has different failure modes
(see [the wrong slot is extracted](#the-wrong-slot-is-extracted)).

**Step 3: confirm training ran.** In the OPM pipeline, training is
triggered by `mycroft.ready`. If your bus doesn't emit that event, the
first inference will train lazily — but bus subscription order can mean
the first inference uses an empty engine. Force a `train()` after
registration completes.

## Confidence is too low

The right intent wins `predict(top_n=...)` but `conf` is below your
configured `conf_high`/`conf_med`.

**Step 1: check template coverage.** If you registered samples like
`"play {song}"` but no entity samples, the template matcher contributes
nothing and the SVM is on its own. Add `entity_samples={"song": [...]}` so
templates expand into real training utterances.

**Step 2: check the −25% template penalty.** If your intent *has* templates
and none of them fuzzy-match the utterance, conf is multiplied by 0.75
— a 0.85 raw score becomes 0.64. Either add more template variants or
accept the penalty as a useful signal that the utterance is unusual.

**Step 3: add paraphrases.** The SVM is generalizing from a bag-of-words
over your samples. If users say "could you possibly turn on" and your
samples only have "turn on", you're missing "could", "possibly". Add more
diverse samples or shorter templates.

## The wrong intent fires

A different intent wins the prediction.

**Step 1: token overlap.** Print both intents' samples and look for shared
high-signal tokens. If "play music" and "stop music" both contain "music",
the SVM's bag-of-words view treats them similarly.

**Step 2: hard-negatives.** Try registering the *intended* wrong utterances
as samples of the *intended right* intent. Counter-intuitively, this makes
the hard-negative mining in `add_negative` work harder for the competing
intent.

**Step 3: structural limit.** If two intents differ only in word order,
linha-fina cannot tell them apart. Pick a matcher with positional features
(transformer-based, or padatious-rules with explicit templates).

## The wrong slot is extracted

The intent is right but `slots["song"]` is empty or contains junk.

**Step 1: keyword vocabulary.** Slot values are extracted by literal
keyword match against `entity_samples`. If the user said "africa" but you
registered `["Africa - Toto"]`, the literal match fails. Register the
shortest unique form.

**Step 2: 3-char minimum.** Keywords shorter than 3 characters are silently
skipped. If your entity values include short codes (state abbreviations,
2-letter country codes) they won't match. Fork `keywords.py` or pad the
values.

**Step 3: template precedence.** Templates extract slots first, keywords as
fallback. If the template fuzzy-matched but the slot span is wrong, the
template wins and the keyword extraction is discarded for that slot.
Inspect with:

```python
from linha_fina.templates import TemplateMatcher
tm = TemplateMatcher()
tm.add_templates(["play {song}"])
print(tm.match("play africa now"))
# May extract {'song': 'africa now'} instead of {'song': 'africa'}
```

Tighter templates (`"play {song}"` vs `"play {song} now"`) help.

## False positives on chitchat

The engine fires confidently on utterances that have nothing to do with
any registered intent.

**Step 1: raise `conf_high`.** If linha-fina is one of several pipeline
plugins, a stricter threshold lets a better-suited engine downstream win.

**Step 2: structural again.** Bag-of-words plus one-vs-rest is inherently
prone to false positives on unfamiliar inputs — every unfamiliar utterance
gets scored against every intent and the highest noise wins. This is one
of the reasons linha-fina is a baseline, not a production recommendation.

## Training is slow

`train()` takes seconds on a small intent set.

**Step 1: parallelism.** `DynamicClassifier.train()` already uses a
`ThreadPoolExecutor`. If you're on a constrained environment (single-core
container) you'll see linear scaling in intent count.

**Step 2: vocabulary size.** Featurization runs over the full keyword
vocabulary for every training sample. If you've registered tens of
thousands of entity values, switch to the Aho-Corasick backend (see
[Tuning](tuning.md)).

**Step 3: lazy vs eager.** If you're calling `register_intent` in a loop
with `instant_train=True`, you're retraining the whole stack on every call.
Use the default lazy mode and `train()` once at the end.

## Plugin isn't loaded by OVOS

You added `ovos-linha-fina-pipeline-plugin` to your config but the intent
service never invokes it.

**Step 1: install location.** OPM discovers plugins via entry points. After
`pip install linha-fina`, confirm with:

```bash
python -c "from importlib.metadata import entry_points; \
           print([e.name for e in entry_points(group='opm.pipeline')])"
```

`ovos-linha-fina-pipeline-plugin` should appear.

**Step 2: pipeline order.** The `intents.pipeline` array determines which
plugins run and in what order. If a higher-priority plugin always wins,
linha-fina never gets called.

**Step 3: language.** The pipeline only registers languages listed in
`lang` + `secondary_langs`. If a skill registers intents for `de-DE` but
OVOS isn't configured for German, those intents go nowhere.

---
[← Tuning](tuning.md) · [Home](index.md)
