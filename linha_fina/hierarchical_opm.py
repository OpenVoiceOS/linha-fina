"""Hierarchical OPM pipeline wrapping :class:`HierarchicalIntentEngine`.

This pipeline routes adds/removes by the ``skill_id`` extracted from the
intent label (``"skill_id:intent_name"``). Each ``skill_id`` becomes a
*domain* with its own per-intent SVM engine. At inference time a top-level
classifier picks one domain and only that domain's engine is scored.

linha-fina's per-intent SVM needs at least 3 intents *globally* before
training succeeds (its own constraint, unchanged from the flat pipeline),
and the top-level domain classifier needs at least 2 domains before it can
train.
"""

from typing import Dict, List, Optional, Union

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_config.config import Configuration
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from linha_fina.hierarchical_engine import HierarchicalIntentEngine
from linha_fina.opm import LinhaFinaPipeline, LinhaFinaIntent


def _split_intent_label(label: str):
    """Split ``skill_id:intent_name`` into ``(skill_id, intent_name)``."""
    if ":" in label:
        skill_id, intent_name = label.split(":", 1)
        return skill_id, intent_name
    return label, label


class HierarchicalLinhaFinaPipeline(LinhaFinaPipeline):
    """Two-stage LinhaFina pipeline using hierarchical domain routing.

    Intents are partitioned by ``skill_id`` (the prefix before the ``:``
    in the intent label) into per-skill :class:`IntentEngine` instances.
    A top-level classifier maps each query to a single ``skill_id`` and
    only that skill's engine resolves the intent. Utterances whose best
    domain scores below ``domain_threshold`` are rejected before any
    sub-engine runs.

    Configuration is read from ``intents.linha_fina_hierarchical`` so this
    plugin can coexist with the flat and domain plugins in the same OVOS
    instance. Accepts every key the flat plugin does, plus
    ``domain_threshold`` — the minimum top-level classifier confidence
    required to route a query (``0.0`` disables the gate).
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        if config is None:
            config = Configuration().get("intents", {}).get("linha_fina_hierarchical", {})
        # set before super().__init__ — _make_engine() runs inside it
        self.domain_threshold = (config or {}).get("domain_threshold", 0.0)
        super().__init__(bus=bus, config=config or {})

    # ── hook overrides ────────────────────────────────────────────────────
    def _make_engine(self) -> HierarchicalIntentEngine:
        return HierarchicalIntentEngine(domain_threshold=self.domain_threshold)

    def _add_intent(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: HierarchicalIntentEngine = self.containers[lang]
        engine.register_domain_intent(skill_id, name, samples)

    def _add_entity(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: HierarchicalIntentEngine = self.containers[lang]
        engine.register_domain_entity(skill_id, name, samples)

    def _remove_intent(self, intent_name: str) -> None:
        skill_id, _ = _split_intent_label(intent_name)
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain_intent(skill_id, intent_name)
            except Exception as e:
                LOG.debug(f"remove_domain_intent({skill_id},{intent_name}): {e}")

    def _remove_entity(self, name: str, lang: str) -> None:
        # entities are scoped per-domain inside HierarchicalIntentEngine; the
        # underlying engines drop them with the domain on remove_domain.
        # individual removal is a no-op here.
        return

    def _remove_skill(self, skill_id: str) -> None:
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain(skill_id)
            except Exception as e:
                LOG.debug(f"remove_domain({skill_id}): {e}")
        # update registered_intents bookkeeping
        keep = [i for i in self.registered_intents
                if _split_intent_label(i)[0] != skill_id]
        self.registered_intents[:] = keep

    # ── inference ─────────────────────────────────────────────────────────
    def calc_intent(self, utterances, lang: str = None,
                    message: Optional[Message] = None) -> Optional[LinhaFinaIntent]:
        if isinstance(utterances, str):
            utterances = [utterances]
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            return None
        lang = self._get_closest_lang(lang or self.lang)
        if lang is None:
            return None
        engine: HierarchicalIntentEngine = self.containers[lang]
        if not engine.domains:
            return None
        best: Optional[LinhaFinaIntent] = None
        for utt in utterances:
            try:
                m = engine.calc_intent(utt)
            except Exception as e:
                LOG.error(f"HierarchicalLinhaFina calc_intent error: {e}")
                continue
            if m is None or m.name is None:
                continue
            if best is None or m.conf > best.conf:
                best = LinhaFinaIntent(sent=utt, name=m.name,
                                       conf=m.conf, matches=m.slots)
        return best
