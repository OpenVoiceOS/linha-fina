# Domain LinhaFina pipeline

The `DomainLinhaFinaPipeline` is a domain-partitioned variant of the flat
[`LinhaFinaPipeline`](index.md). Intents are partitioned by ``skill_id``
(the prefix before the ``:`` in the intent label) into per-skill
[`IntentEngine`](index.md) instances. At query time every per-skill
engine produces a candidate `IntentMatch` and the global ``argmax`` by
confidence wins (adapt's classic parallel-argmax shape).

This narrows the negative-sample pool each per-intent SVM trains
against, so SVMs see other intents from the **same skill** instead of
every intent in the system — boosting precision when many skills share
overlapping vocabulary.

## OPM entry point

```yaml
opm.pipeline:
  ovos-linha-fina-domain-pipeline-plugin = linha_fina.opm:DomainLinhaFinaPipeline
```

Both pipelines are shipped from the same package; pick one in your OVOS
pipeline configuration.

## Configuration

Under `intents` / your pipeline section in `mycroft.conf`:

```json
{
  "linha_fina_domain": {
    "conf_high": 0.8,
    "conf_med": 0.6,
    "conf_low": 0.4
  }
}
```

| key                       | default | meaning                                                                 |
| ------------------------- | ------- | ----------------------------------------------------------------------- |
| `conf_high`               | 0.8     | threshold for `match_high`                                              |
| `conf_med`                | 0.6     | threshold for `match_medium`                                            |
| `conf_low`                | 0.4     | threshold for `match_low`                                               |

## Routing

The intent label coming over the OVOS bus
(``padatious:register_intent``) is expected to be
``skill_id:intent_name``. The pipeline:

- extracts ``skill_id`` from the label,
- routes ``register_intent`` / ``detach_intent`` / ``detach_skill`` to
  the matching domain engine,
- at inference time runs every per-skill engine and returns the global
  argmax by confidence.

## SVM training: ≥3 intents *globally*

linha-fina trains a **per-intent SVM**, drawing negative samples from
**other intents in the same engine**. With fewer than 3 intents the
engine cannot build a valid negative set and training raises.

Because there is **no top-level router** any more, there is no
per-domain minimum — a skill that exposes a single intent is still
perfectly matchable as long as the corpus as a whole has enough intents
to train the SVMs. The constraint is identical to the flat
`LinhaFinaPipeline`: ≥3 intents in total.

---
[← OPM pipeline](pipeline.md) · [Home](index.md) · [Tuning →](tuning.md)
