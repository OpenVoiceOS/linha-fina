"""Domain-aware intent engine for hierarchical intent organisation.

Mirrors the API shipped by `nebulento.DomainIntentContainer`,
`ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`,
and `padacioso.DomainIntentContainer`: intents are grouped into *domains*,
each domain owns its own :class:`~linha_fina.engine.IntentEngine`, and
queries are resolved by parallel-argmax across every per-domain engine
(adapt's classic shape).

For Linha Fina specifically the partitioning still matters — per-intent
SVMs use the rest of the corpus as negative samples, so isolating intents
into per-domain engines means the SVMs see more relevant negatives
(other intents in the same domain) and lighter overall negative-set
imbalance.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from linha_fina.engine import IntentEngine, IntentMatch

LOG = logging.getLogger(__name__)


class DomainIntentEngine:
    """Domain-partitioned intent engine using parallel-argmax routing.

    Intents are grouped into *domains*. Each domain owns its own
    :class:`IntentEngine`. At query time every domain engine produces a
    candidate :class:`IntentMatch` and the global ``argmax`` by
    ``conf`` wins. Domains can be selected explicitly to scope inference
    to a single sub-engine.

    Example::

        from linha_fina import DomainIntentEngine

        d = DomainIntentEngine()
        d.register_domain_intent("media", "play",
                                  ["play {song}", "put on {song}"],
                                  entity_samples={"song": ["africa", "hey jude"]})
        d.register_domain_intent("home", "lights_on",
                                  ["turn on the lights", "lights on"])

        match = d.calc_intent("play africa")
        # match.name == "play"

    Args:
        instant_train: Forwarded to every :class:`IntentEngine` created
            internally.  When ``True``, the engine retrains on every
            registration; when ``False`` (default), training is deferred
            until the first prediction.
    """

    def __init__(self, instant_train: bool = False) -> None:
        self.instant_train = instant_train
        #: Per-domain intent engines, keyed by domain name.
        self.domains: Dict[str, IntentEngine] = {}
        #: Raw training samples per (domain, intent) for inspection / re-training.
        self.training_data: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)

    def train(self) -> None:
        """Train every per-domain :class:`IntentEngine`.

        Mirrors :meth:`IntentEngine.train`, so callers (e.g. the
        ``mycroft.ready`` initial-train hook in ``LinhaFinaPipeline``) can
        eagerly pre-train a :class:`DomainIntentEngine` the same way they
        pre-train a flat one, instead of relying on the lazy
        train-on-first-prediction behaviour of the underlying engines.
        """
        for engine in self.domains.values():
            engine.train()

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                samples: List[str],
                                entity_samples: Optional[Dict[str, List[str]]] = None) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`IntentEngine` on first use.

        Args:
            domain_name: Target domain (created if it does not exist).
            intent_name: Unique intent name within the domain.
            samples: Training utterances for the intent. May contain
                ``{slot}`` placeholders.
            entity_samples: Optional mapping of slot name to example values,
                forwarded directly to :meth:`IntentEngine.register_intent`.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentEngine(instant_train=self.instant_train)
        self.domains[domain_name].register_intent(
            intent_name, samples, entity_samples=entity_samples
        )
        self.training_data[domain_name][intent_name] = list(samples)

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove an intent from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)
        self.training_data.get(domain_name, {}).pop(intent_name, None)

    def register_domain_entity(self, domain_name: str, entity_name: str,
                                samples: List[str],
                                intent_name: Optional[str] = None) -> None:
        """Register entity samples inside a domain.

        Args:
            domain_name: The domain whose engine should register the entity.
            entity_name: Slot / entity name.
            samples: Example values for the slot.
            intent_name: If provided, scope the entity to a single intent;
                otherwise share it across every intent currently registered
                in the domain.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentEngine(instant_train=self.instant_train)
        self.domains[domain_name].register_entity(
            entity_name, samples, intent_name=intent_name
        )

    # ── query API ──────────────────────────────────────────────────────────

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> IntentMatch:
        """Return the best intent match for *query*.

        Args:
            query: The utterance to match.
            domain: If given, scope inference to this domain's engine only.

        Returns:
            The best :class:`IntentMatch` across the union of per-domain
            predictions (or just this domain when ``domain`` is given).
            If nothing matched, returns ``IntentMatch(name=None, slots={},
            conf=0.0)``.
        """
        if domain is not None:
            if domain in self.domains:
                return self.domains[domain].calc_intent(query)
            return IntentMatch(name=None, slots={}, conf=0.0)

        best = IntentMatch(name=None, slots={}, conf=0.0)
        for name, engine in self.domains.items():
            try:
                m = engine.calc_intent(query)
            except Exception as e:
                LOG.warning(f"calc_intent failed for domain {name!r}: {e}")
                continue
            if m is None or m.name is None:
                continue
            if m.conf > best.conf:
                best = m
        return best

    def predict(self, query: str,
                 domain: Optional[str] = None,
                 top_n: int = 3) -> List[IntentMatch]:
        """Return the top ``top_n`` intents across the union of domains.

        Args:
            query: The utterance to match.
            domain: If given, scope to this domain only.
            top_n: Number of top matches to return.
        """
        if domain is not None:
            if domain in self.domains:
                return self.domains[domain].predict(query, top_n=top_n)
            return [IntentMatch(name=None, slots={}, conf=0.0)]

        candidates: List[IntentMatch] = []
        for name, engine in self.domains.items():
            try:
                candidates.extend(engine.predict(query, top_n=top_n))
            except Exception as e:
                LOG.warning(f"predict failed for domain {name!r}: {e}")
                continue
        candidates = [c for c in candidates if c is not None and c.name is not None]
        if not candidates:
            return [IntentMatch(name=None, slots={}, conf=0.0)]
        candidates.sort(key=lambda m: m.conf, reverse=True)
        return candidates[:top_n]
