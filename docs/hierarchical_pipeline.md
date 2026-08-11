# Hierarchical LinhaFina pipeline

The `HierarchicalLinhaFinaPipeline` is a two-stage variant of the flat
[`LinhaFinaPipeline`](index.md). Intents are partitioned by ``skill_id``
(the prefix before the ``:`` in the intent label) into per-skill
[`IntentEngine`](index.md) instances. A top-level classifier maps each
query to a single ``skill_id``, and only that skill's engine is scored to
resolve the intent.

The top-level domain classifier is itself an `IntentEngine`, trained with
each ``skill_id`` as a label and the union of that skill's intent samples
as that label's training utterances. Two-stage routing keeps every
per-intent SVM small — its negatives are only the other intents in the
same skill — and moves skill disambiguation into a dedicated classifier
instead of a global argmax.

## OPM entry point

```yaml
opm.pipeline:
  ovos-linha-fina-hierarchical-pipeline-plugin = linha_fina.hierarchical_opm:HierarchicalLinhaFinaPipeline
```

All pipelines are shipped from the same package; pick one in your OVOS
pipeline configuration.

## Configuration

Under `intents` / your pipeline section in `mycroft.conf`:

```json
{
  "linha_fina_hierarchical": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4,
    "domain_threshold": 0.0
  }
}
```

| key                | default | meaning                                                              |
| ------------------ | ------- | -------------------------------------------------------------------- |
| `conf_high`        | 0.8     | threshold for `match_high`                                           |
| `conf_med`         | 0.6     | threshold for `match_medium`                                         |
| `conf_low`         | 0.4     | threshold for `match_low`                                            |
| `domain_threshold` | 0.0     | minimum top-level classifier confidence required to route a query    |

## Routing

The intent label coming over the OVOS bus
(``padatious:register_intent``) is expected to be
``skill_id:intent_name``. The pipeline:

- extracts ``skill_id`` from the label,
- routes ``register_intent`` / ``detach_intent`` / ``detach_skill`` to
  the matching domain engine,
- at inference time runs the top-level classifier to pick one
  ``skill_id``, then scores only that skill's per-intent SVM engine.

## Off-topic rejection

`domain_threshold` gates routing. When the top-level classifier's best
domain scores below the threshold, the pipeline returns a no-match
without running any per-skill engine. The default ``0.0`` disables the
gate, so every query is routed to its best domain.

## SVM training: ≥3 intents *globally*, ≥2 domains

linha-fina trains a **per-intent SVM**, drawing negative samples from
**other intents in the same engine**. With fewer than 3 intents the
engine cannot build a valid negative set and training raises — the same
constraint as the flat `LinhaFinaPipeline`.

The top-level domain classifier is an `IntentEngine` too, so it needs at
least 2 domains before it can train and route queries.
