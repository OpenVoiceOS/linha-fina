"""Hierarchical intent engine — two-stage domain routing.

Mirrors the API shipped by `nebulento.HierarchicalIntentContainer`:
intents are grouped into *domains*, a top-level classifier picks one
domain per query, and only that domain's :class:`~linha_fina.engine.IntentEngine`
is scored.

For Linha Fina the top-level classifier is itself an
:class:`~linha_fina.engine.IntentEngine`, trained with each domain name as
the label and the union of that domain's intent samples as that label's
training utterances. This keeps every per-domain SVM small — its negatives
are only the other intents in the same domain — and pushes domain
disambiguation into a dedicated classifier rather than a global argmax.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from linha_fina.engine import IntentEngine, IntentMatch


class HierarchicalIntentEngine:
    """Two-stage intent engine: domain classification followed by intent matching.

    Intents are grouped into *domains*. Each domain owns its own
    :class:`IntentEngine`. At query time the engine first selects the most
    likely domain via :attr:`domain_engine`, then runs that domain's engine
    to find the best intent within it.

    The top-level domain classifier is trained automatically: every sample
    passed to :meth:`register_domain_intent` is also fed to
    :attr:`domain_engine` under its domain name, so the engine works
    standalone with no manual classifier setup.

    Domains can also be selected explicitly, bypassing the top-level
    classifier.

    Example::

        from linha_fina import HierarchicalIntentEngine

        d = HierarchicalIntentEngine()
        d.register_domain_intent("media", "play", ["play music", "put on a song"])
        d.register_domain_intent("home", "lights_on", ["lights on", "turn on the lights"])

        match = d.calc_intent("play some jazz")
        # match.name == "play"

    Args:
        instant_train: Forwarded to every :class:`IntentEngine` created
            internally (the per-domain engines and the top-level
            classifier). When ``True``, engines retrain on every
            registration; when ``False`` (default), training is deferred
            until the first prediction.
        domain_threshold: Minimum confidence the top-level classifier must
            reach for a query to be routed at all. When the best domain
            scores below this, :meth:`calc_intent` returns a no-match
            instead of resolving an intent — this is the off-topic
            rejection gate. ``0.0`` (default) disables the gate; every
            query is routed to its best domain.
    """

    def __init__(self, instant_train: bool = False,
                 domain_threshold: float = 0.0) -> None:
        self.instant_train = instant_train
        self.domain_threshold = domain_threshold
        #: Top-level classifier that maps free-text queries to a domain name.
        self.domain_engine: IntentEngine = IntentEngine(instant_train=instant_train)
        #: Per-domain intent engines, keyed by domain name.
        self.domains: Dict[str, IntentEngine] = {}
        #: Raw training samples per (domain, intent) for inspection / re-training.
        self.training_data: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        #: Domains whose classifier entry is stale and must be rebuilt before a query.
        self._dirty_domains: set = set()

    # ── internal ───────────────────────────────────────────────────────────

    def _sync_domain_classifier(self) -> None:
        """Rebuild stale classifier entries.

        Registration only marks a domain dirty; the top-level classifier is
        rebuilt here, lazily, the first time a query needs it. This keeps
        bulk registration linear instead of re-expanding the whole corpus
        per call.
        """
        for domain_name in self._dirty_domains:
            self.domain_engine.remove_intent(domain_name)
            samples: List[str] = []
            for intent_samples in self.training_data.get(domain_name, {}).values():
                samples += intent_samples
            if samples:
                self.domain_engine.register_intent(domain_name, samples)
        self._dirty_domains.clear()

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents, entities, and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        self._dirty_domains.discard(domain_name)
        self.domain_engine.remove_intent(domain_name)

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                samples: List[str],
                                entity_samples: Optional[Dict[str, List[str]]] = None) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`IntentEngine` on first use. The
        top-level domain classifier is marked stale and rebuilt lazily on
        the next query.

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
        self._dirty_domains.add(domain_name)

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove an intent from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)
        self.training_data.get(domain_name, {}).pop(intent_name, None)
        self._dirty_domains.add(domain_name)

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

    def remove_domain_entity(self, domain_name: str, entity_name: str,
                              intent_name: Optional[str] = None) -> None:
        """Remove an entity from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name, intent_name=intent_name)

    # ── query API ──────────────────────────────────────────────────────────

    def calc_domain(self, query: str) -> IntentMatch:
        """Classify *query* into the best-matching domain.

        Args:
            query: Raw utterance to classify.

        Returns:
            An :class:`IntentMatch` whose ``name`` is the predicted domain
            name (or ``None`` if no domain matched).
        """
        self._sync_domain_classifier()
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> IntentMatch:
        """Return the best intent match for *query*, optionally within *domain*.

        If *domain* is ``None``, the domain is inferred by
        :meth:`calc_domain`. When the inferred domain scores below
        :attr:`domain_threshold`, or the inferred/supplied domain has no
        registered intents, a no-match result is returned. Passing *domain*
        explicitly bypasses the classifier and the threshold gate.

        Args:
            query: The utterance to match.
            domain: Domain to restrict matching to. ``None`` triggers
                automatic domain classification.

        Returns:
            The best :class:`IntentMatch`, or ``IntentMatch(name=None,
            slots={}, conf=0.0)`` when no domain or intent could be matched.
        """
        no_match = IntentMatch(name=None, slots={}, conf=0.0)

        resolved_domain = domain
        if resolved_domain is None:
            self._sync_domain_classifier()
            try:
                dom_result = self.domain_engine.calc_intent(query)
            except Exception:
                return no_match
            if dom_result is None or dom_result.name is None:
                return no_match
            if dom_result.conf < self.domain_threshold:
                return no_match
            resolved_domain = dom_result.name

        if resolved_domain in self.domains:
            try:
                return self.domains[resolved_domain].calc_intent(query)
            except Exception:
                return no_match
        return no_match

    def predict(self, query: str,
                 domain: Optional[str] = None,
                 top_n: int = 3) -> List[IntentMatch]:
        """Return the top ``top_n`` intents from the classified (or given) domain.

        Args:
            query: The utterance to match.
            domain: If given, scope to this domain only, bypassing the
                top-level classifier and threshold gate.
            top_n: Number of top matches to return.
        """
        no_match = [IntentMatch(name=None, slots={}, conf=0.0)]

        resolved_domain = domain
        if resolved_domain is None:
            self._sync_domain_classifier()
            try:
                dom_result = self.domain_engine.calc_intent(query)
            except Exception:
                return no_match
            if dom_result is None or dom_result.name is None:
                return no_match
            if dom_result.conf < self.domain_threshold:
                return no_match
            resolved_domain = dom_result.name

        if resolved_domain in self.domains:
            try:
                return self.domains[resolved_domain].predict(query, top_n=top_n)
            except Exception:
                return no_match
        return no_match
