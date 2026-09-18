# Concepts

## What is intent matching?

A voice assistant turns a user utterance — "play africa", "what's the weather
in Paris?", "turn off the kitchen lights" — into two things:

1. an **intent**: a label naming what the user wants (`play_song`, `weather`,
   `lights_off`)
2. **slots**: structured arguments extracted from the utterance (`{song:
   "africa"}`, `{city: "Paris"}`, `{room: "kitchen"}`)

A skill registers a handful of *example utterances* per intent. The intent
matcher's job at runtime is: given a new utterance, decide which registered
intent (if any) it matches, and pull out the slot values.

This is harder than it sounds. Users phrase the same request a hundred
different ways. The matcher needs to generalize beyond the literal samples
without firing on unrelated input ("what time is it?" should *not* match
`play_song`, no matter how many ways you ask for music).

## Where linha-fina sits

`linha-fina` is a baseline implementation: it borrows padatious's overall
architecture (template expansion → bag-of-keywords features → per-intent
classifier) and substitutes a small one-vs-rest SVM stack for padatious's
neural network. It exists so the benchmarks have a "what if you just used
classical sklearn?" reference point to compare padatious and its newer
replacements against. It is not tuned for production accuracy or latency.

## Three families of intent matcher

Most matchers fall into one of three buckets, each with sharp tradeoffs:

| Family | Example | How it works | Wins at | Loses at |
|---|---|---|---|---|
| **Pattern / template** | padacioso, padatious-rules | Glob or regex-like patterns with `{slot}` placeholders | Exact slot extraction, transparent behaviour | Paraphrases ("can you put on…" vs "play…") |
| **Statistical classifier** | padatious (TF-IDF + NN), nebulento | Train a model on labelled utterances; predict at runtime | Paraphrases, fuzzy phrasing | Brittle slot extraction, needs ≥3 intents |
| **Keyword / vocab** | adapt, ovos-adapt-pipeline | Match on presence of registered keywords | Fast, deterministic, works with 1 sample | Word-salad false positives |

`linha-fina` combines all three. The classifier picks candidate intents, the
template matcher refines the score and extracts slots, the keyword extractor
fills slots the template missed.

## The linha-fina pipeline

```
                ┌─────────────────────────────────────────────────┐
                │  utterance: "play africa"                       │
                └───────────────────┬─────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────────┐
            │ 1. KeywordFeatures.one_hot_encode(utterance)      │
            │    → binary vector over the training vocabulary   │
            └───────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────────┐
            │ 2. DynamicClassifier.predict(vector)              │
            │    → one binary SVM per intent (one-vs-rest)      │
            │    → {play: 0.82, stop: 0.04, next: 0.11}         │
            └───────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────────┐
            │ 3. For each candidate intent:                     │
            │    a. KeywordFeatures.extract → {song: "africa"}  │
            │       if entities matched: conf *= 1.10           │
            │    b. TemplateMatcher.match  → {song: "africa"}   │
            │       if templates exist but none matched:        │
            │         conf *= 0.75                              │
            └───────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────────┐
            │ IntentMatch(name="play", slots={"song":"africa"}, │
            │             conf=0.90)                            │
            └───────────────────────────────────────────────────┘
```

Each layer compensates for the others:

- The **SVM** generalizes across phrasings the user invented that aren't in
  the samples ("put on", "I want to hear", "let's listen to").
- The **template matcher** anchors the score: a high-SVM-confidence prediction
  that *can* be expressed as a known template is almost certainly right; one
  that can't is suspicious.
- The **keyword extractor** is the slot-extraction safety net when neither
  template nor model can pinpoint a span.

## Why one-vs-rest SVMs?

`DynamicClassifier` trains one binary classifier per intent instead of one
multi-class model. For each intent it builds:

- **positives**: the registered sample utterances for that intent
- **negatives**: sample utterances from *other* intents in the engine, capped
  at `3 × len(positives)` and selected for similarity to the positives so the
  SVM learns the hardest boundary

This architecture has three useful properties:

1. **Dynamic registration.** Adding a new intent doesn't invalidate the
   already-trained classifiers — only the affected one (and the negatives
   pulled into others) needs to retrain. Useful in OVOS where skills load
   and unload at runtime.
2. **Better signal per classifier.** A single multi-class SVM with 50 intents
   has to learn a 50-way decision surface from very few samples. One-vs-rest
   binary SVMs each solve a much easier 2-class problem.
3. **Confidence per intent.** Each binary classifier produces its own
   probability, so the engine can report calibrated per-intent confidences
   instead of softmax probabilities over a single model.

The cost is that scoring an utterance against N intents requires N model
evaluations. In practice this is cheap (each model is tiny) but it does mean
the engine scales linearly in registered intents per language.

## What linha-fina is *not* good at

- **Long, syntactically complex sentences.** Features are bag-of-words; word
  order is invisible to the SVM. "open the door before closing the window"
  and "close the door before opening the window" look identical.
- **Single-sample intents in isolation.** The SVM needs ≥3 intents registered
  in total before it can train. With one or two intents you'll get
  template-only matching.
- **Slot values not seen at training time.** `KeywordFeatures` matches
  literally against the strings you register. For open-ended slots (free text
  song titles, novel city names) you need a downstream NER step, or rely on
  the template matcher to capture the slot span.
- **High-throughput streaming.** Lazy training means the first inference
  after a registration pays a training cost. Call `train()` explicitly during
  warmup if latency on the first request matters.

See [Tuning](tuning.md) for knobs that mitigate some of these.

---
[Home](index.md) · [Quickstart →](quickstart.md)
