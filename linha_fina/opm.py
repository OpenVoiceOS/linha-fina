"""Intent service wrapping LinhaFina."""

from functools import lru_cache
from os.path import isfile
from typing import Optional, Dict, List, Union

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch
from ovos_spec_tools import SpecMessage, closest_lang, standardize_lang
from ovos_utils import flatten_list
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from linha_fina.domain_engine import DomainIntentEngine
from linha_fina.engine import IntentEngine


class LinhaFinaIntent:
    """
    A set of data describing how a query fits into an intent
    Attributes:
        name (str): Name of matched intent
        sent (str): The input utterance associated with the intent
        conf (float): Confidence (from 0.0 to 1.0)
        matches (dict of str -> str): Key is the name of the entity and
            value is the extracted part of the sentence
    """

    def __init__(self, name, sent, matches=None, conf=0.0):
        self.name = name
        self.sent = sent
        self.matches = matches or {}
        self.conf = conf

    def __getitem__(self, item):
        return self.matches.__getitem__(item)

    def __contains__(self, item):
        return self.matches.__contains__(item)

    def get(self, key, default=None):
        return self.matches.get(key, default)

    def __repr__(self):
        return repr(self.__dict__)


class LinhaFinaPipeline(ConfidenceMatcherPipeline):
    """Service class for LinhaFina intent matching."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        super().__init__(config=config or {}, bus=bus)

        core_config = Configuration()
        self.lang = standardize_lang(core_config.get("lang", "en-US"))
        langs = core_config.get('secondary_langs') or []
        if self.lang not in langs:
            langs.append(self.lang)
        langs = [standardize_lang(l) for l in langs]
        self.conf_high = self.config.get("conf_high") or 0.8
        self.conf_med = self.config.get("conf_med") or 0.6
        self.conf_low = self.config.get("conf_low") or 0.4

        self.containers = {lang: self._make_engine() for lang in langs}

        # legacy (padatious-compatible) registration surface
        self.bus.on('padatious:register_intent', self.register_intent)
        self.bus.on('padatious:register_entity', self.register_entity)
        self.bus.on('detach_intent', self.handle_detach_intent)
        self.bus.on('detach_skill', self.handle_detach_skill)
        self.bus.on('mycroft.ready', self.handle_initial_train)

        # OVOS-INTENT-4 registration surface (alongside the legacy one).
        # LinhaFina is a sample/template matcher, so it consumes the template
        # registration topic (§6) but NOT the keyword topic (§5/§11).
        self.bus.on(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                    self.handle_register_template)
        self.bus.on(SpecMessage.ENTITY_REGISTER.value,
                    self.handle_register_entity_spec)
        self.bus.on(SpecMessage.INTENT_DEREGISTER.value,
                    self.handle_deregister_intent_spec)
        self.bus.on(SpecMessage.ENTITY_DEREGISTER.value,
                    self.handle_deregister_entity_spec)
        self.bus.on(SpecMessage.SKILL_DEREGISTER.value,
                    self.handle_deregister_skill_spec)
        self.bus.on(SpecMessage.INTENT_ENABLE.value,
                    self.handle_enable_intent_spec)
        self.bus.on(SpecMessage.INTENT_DISABLE.value,
                    self.handle_disable_intent_spec)

        self.registered_intents = []
        self.registered_entities = []
        # INTENT-4 §8.5 — intents disabled without losing their definition;
        # excluded from match candidacy until re-enabled.
        self.disabled_intents: set = set()
        self.max_words = 50  # if an utterance contains more words than this, don't attempt to match
        LOG.debug('Loaded LinhaFina intent parser.')

    # ── hooks (override in subclasses) ────────────────────────────────────
    def _make_engine(self):
        """Build the per-language intent engine. Override to swap engines."""
        return IntentEngine()

    def _add_intent(self, lang: str, name: str, samples: List[str]) -> None:
        """Register an intent into the engine for *lang*."""
        self.containers[lang].register_intent(name, samples)

    def _add_entity(self, lang: str, name: str, samples: List[str]) -> None:
        """Register an entity into the engine for *lang*."""
        self.containers[lang].register_entity(name, samples)

    def _remove_intent(self, intent_name: str) -> None:
        """Remove an intent from every per-language engine."""
        for lang in self.containers:
            self.containers[lang].remove_intent(intent_name)

    def _remove_entity(self, name: str, lang: str) -> None:
        if lang in self.containers:
            self.containers[lang].remove_entity(name)

    def _remove_skill(self, skill_id: str) -> None:
        """Remove every intent/entity belonging to *skill_id*."""
        skill_id_colon = skill_id + ":"
        remove_list = [i for i in self.registered_intents if i.startswith(skill_id_colon)]
        for i in remove_list:
            self._detach_intent(i)
        for en in self.registered_entities:
            if en["name"].startswith(skill_id_colon):
                self._remove_entity(en["name"], en["lang"])

    def handle_initial_train(self, message: Message):
        # otherwise training happens on first inference
        for lang in self.containers:
            self.containers[lang].train()

    def _match_level(self, utterances, limit, lang=None,
                     message: Optional[Message] = None) -> Optional[IntentHandlerMatch]:
        """Match intent and make sure a certain level of confidence is reached.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
            limit (float): required confidence level.
        """
        LOG.debug(f'LinhaFina Matching confidence > {limit}')
        # call flatten in case someone is sending the old style list of tuples
        utterances = flatten_list(utterances)
        lang = standardize_lang(lang or self.lang)
        lf_intent = self.calc_intent(utterances, lang, message)
        if lf_intent is not None and lf_intent.conf > limit:
            skill_id = lf_intent.name.split(':')[0]
            return IntentHandlerMatch(match_type=lf_intent.name,
                                      match_data=lf_intent.matches,
                                      skill_id=skill_id,
                                      utterance=lf_intent.sent)

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for high confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for medium confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for low confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_low, lang, message)

    def _detach_intent(self, intent_name):
        """ Remove an intent if it has been registered."""
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            self._remove_intent(intent_name)

    # back-compat alias
    __detach_intent = _detach_intent

    def handle_detach_intent(self, message):
        """Messagebus handler for detaching LinhaFina intent."""
        self._detach_intent(message.data.get('intent_name'))

    def handle_detach_skill(self, message):
        """Messagebus handler for detaching all intents for skill."""
        skill_id = message.data['skill_id']
        self._remove_skill(skill_id)

    @staticmethod
    def _extract_samples(message, object_name):
        """Generic method for registering a LinhaFina object.

        Args:
            message (Message): trigger for action
            object_name (str): type of entry to register
            register_func (callable): function to call for registration
        """
        file_name = message.data.get('file_name')
        samples = message.data.get("samples")
        name = message.data['name']

        LOG.debug('Registering LinhaFina ' + object_name + ': ' + name)

        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error('Could not find file ' + str(file_name))
            return None, None

        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines()]

        return name, samples

    def register_intent(self, message):
        """Messagebus handler for registering intents."""
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang(lang)
        if lang in self.containers:
            name, samples = self._extract_samples(message, 'intent')
            if name is None:
                return
            self.registered_intents.append(message.data['name'])
            self._add_intent(lang, name, samples)

    def register_entity(self, message):
        """Messagebus handler for registering entities."""
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang(lang)
        if lang in self.containers:
            name, samples = self._extract_samples(message, 'entity')
            if name is None:
                return
            self.registered_entities.append(message.data)
            self._add_entity(lang, name, samples)

    # ── OVOS-INTENT-4 registration surface ──────────────────────────────────

    @staticmethod
    def _spec_label(message, key: str) -> Optional[str]:
        """Build the internal ``skill_id:<key>`` label from an INTENT-4 payload.

        INTENT-4 carries ``skill_id`` and ``intent_name`` / ``entity_name`` as
        separate fields (§3.2); LinhaFina keys everything on the combined
        ``skill_id:name`` label, matching the legacy padatious convention.
        """
        skill_id = message.data.get("skill_id")
        name = message.data.get(key)
        if not skill_id or not name:
            LOG.warning(f"Ignoring malformed INTENT-4 payload on {message.msg_type!r}: "
                        f"missing skill_id/{key}")
            return None
        return f"{skill_id}:{name}"

    def handle_register_template(self, message):
        """Consume ``ovos.intent.register.template`` (INTENT-4 §6).

        Template intents are LinhaFina's native definition method. The payload
        carries inline ``samples`` (OVOS-INTENT-1 templates); ``blacklist`` is a
        suppression hint LinhaFina does not yet honour and is ignored.
        """
        lang = standardize_lang(message.data.get("lang", self.lang))
        if lang not in self.containers:
            return
        name = self._spec_label(message, "intent_name")
        if name is None:
            return
        samples = message.data.get("samples")
        if not samples:
            LOG.warning(f"Ignoring INTENT-4 template registration for {name!r}: "
                        f"empty samples")
            return
        self.registered_intents.append(name)
        self._add_intent(lang, name, samples)

    def handle_register_entity_spec(self, message):
        """Consume ``ovos.entity.register`` (INTENT-4 §7)."""
        lang = standardize_lang(message.data.get("lang", self.lang))
        if lang not in self.containers:
            return
        name = self._spec_label(message, "entity_name")
        if name is None:
            return
        samples = message.data.get("samples")
        if not samples:
            LOG.warning(f"Ignoring INTENT-4 entity registration for {name!r}: "
                        f"empty samples")
            return
        self.registered_entities.append({"name": name, "lang": lang,
                                         "samples": samples})
        self._add_entity(lang, name, samples)

    def handle_deregister_intent_spec(self, message):
        """Consume ``ovos.intent.deregister`` (INTENT-4 §8.2)."""
        name = self._spec_label(message, "intent_name")
        if name is None:
            return
        self._detach_intent(name)
        self.disabled_intents.discard(name)

    def handle_deregister_entity_spec(self, message):
        """Consume ``ovos.entity.deregister`` (INTENT-4 §8.3)."""
        name = self._spec_label(message, "entity_name")
        if name is None:
            return
        lang = standardize_lang(message.data.get("lang", self.lang))
        self._remove_entity(name, lang)
        self.registered_entities = [
            en for en in self.registered_entities if en.get("name") != name
        ]

    def handle_deregister_skill_spec(self, message):
        """Consume ``ovos.skill.deregister`` (INTENT-4 §8.4)."""
        # payload shape matches detach_skill — reuse the legacy handler
        self.handle_detach_skill(message)

    def handle_enable_intent_spec(self, message):
        """Consume ``ovos.intent.enable`` (INTENT-4 §8.5)."""
        name = self._spec_label(message, "intent_name")
        if name is not None:
            self.disabled_intents.discard(name)

    def handle_disable_intent_spec(self, message):
        """Consume ``ovos.intent.disable`` (INTENT-4 §8.5)."""
        name = self._spec_label(message, "intent_name")
        if name is not None:
            self.disabled_intents.add(name)

    def calc_intent(self, utterances: List[str], lang: str = None,
                    message: Optional[Message] = None) -> Optional[LinhaFinaIntent]:
        """
        Get the best intent match for the given list of utterances. Utilizes a
        thread pool for overall faster execution. Note that this method is NOT
        compatible with LinhaFina, but is compatible with LinhaFina.
        @param utterances: list of string utterances to get an intent for
        @param lang: language of utterances
        @return:
        """
        if isinstance(utterances, str):
            utterances = [utterances]  # backwards compat when arg was a single string
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            LOG.error(f"utterance exceeds max size of {self.max_words} words, skipping LinhaFina match")
            return None

        lang = lang or self.lang

        lang = self._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        sess = SessionManager.get(message)

        # Invalidate the match cache once per call: (de)registrations mutate the
        # engine between calls, and a stale cache entry would keep a removed
        # intent matching.
        _calc_lf_intent.cache_clear()

        intent_container = self.containers.get(lang)
        # Pass the blacklists as hashable frozensets rather than the Session
        # object — ovos-bus-client>=2.5.1a1 makes Session unhashable, which
        # would raise "unhashable type: 'Session'" at the lru_cache key.
        bl_intents = frozenset(sess.blacklisted_intents or [])
        bl_skills = frozenset(sess.blacklisted_skills or [])
        intents = [_calc_lf_intent(utt, intent_container, bl_intents, bl_skills)
                   for utt in utterances]
        # INTENT-4 §8.5 — disabled intents are excluded from match candidacy
        intents = [i for i in intents
                   if i is not None and i.name not in self.disabled_intents]
        # select best
        if intents:
            return max(intents, key=lambda k: k.conf)

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        if self.containers:
            return closest_lang(lang, list(self.containers.keys()))
        return None

    def shutdown(self):
        self.bus.remove('padatious:register_intent', self.register_intent)
        self.bus.remove('padatious:register_entity', self.register_entity)
        self.bus.remove('detach_intent', self.handle_detach_intent)
        self.bus.remove('detach_skill', self.handle_detach_skill)
        self.bus.remove(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                        self.handle_register_template)
        self.bus.remove(SpecMessage.ENTITY_REGISTER.value,
                        self.handle_register_entity_spec)
        self.bus.remove(SpecMessage.INTENT_DEREGISTER.value,
                        self.handle_deregister_intent_spec)
        self.bus.remove(SpecMessage.ENTITY_DEREGISTER.value,
                        self.handle_deregister_entity_spec)
        self.bus.remove(SpecMessage.SKILL_DEREGISTER.value,
                        self.handle_deregister_skill_spec)
        self.bus.remove(SpecMessage.INTENT_ENABLE.value,
                        self.handle_enable_intent_spec)
        self.bus.remove(SpecMessage.INTENT_DISABLE.value,
                        self.handle_disable_intent_spec)


def _split_intent_label(label: str):
    """Split ``skill_id:intent_name`` into ``(skill_id, intent_name)``."""
    if ":" in label:
        skill_id, intent_name = label.split(":", 1)
        return skill_id, intent_name
    return label, label


class DomainLinhaFinaPipeline(LinhaFinaPipeline):
    """Domain-partitioned LinhaFina pipeline.

    Intents are partitioned by ``skill_id`` (the prefix before the ``:``
    in the intent label) into per-skill :class:`IntentEngine` instances
    held inside a single :class:`DomainIntentEngine` per language.
    Inference runs every per-skill engine in parallel and returns the
    global ``argmax`` by confidence.

    Re-uses the entire bus-handling machinery from
    :class:`LinhaFinaPipeline`; only the per-language engine factory and
    the add/remove hooks differ.
    """

    def _make_engine(self) -> DomainIntentEngine:
        return DomainIntentEngine()

    def _add_intent(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: DomainIntentEngine = self.containers[lang]
        engine.register_domain_intent(skill_id, name, samples)

    def _add_entity(self, lang: str, name: str, samples: List[str]) -> None:
        skill_id, _ = _split_intent_label(name)
        engine: DomainIntentEngine = self.containers[lang]
        engine.register_domain_entity(skill_id, name, samples)

    def _remove_intent(self, intent_name: str) -> None:
        skill_id, _ = _split_intent_label(intent_name)
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain_intent(skill_id, intent_name)
            except Exception as e:
                LOG.debug(f"remove_domain_intent({skill_id},{intent_name}): {e}")

    def _remove_entity(self, name: str, lang: str) -> None:
        # entities are scoped per-domain inside DomainIntentEngine; the
        # underlying engines drop them with the domain on remove_domain.
        return

    def _remove_skill(self, skill_id: str) -> None:
        for lang, engine in self.containers.items():
            try:
                engine.remove_domain(skill_id)
            except Exception as e:
                LOG.debug(f"remove_domain({skill_id}): {e}")
        keep = [i for i in self.registered_intents
                if _split_intent_label(i)[0] != skill_id]
        self.registered_intents[:] = keep

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
        engine: DomainIntentEngine = self.containers[lang]
        if not engine.domains:
            return None
        best: Optional[LinhaFinaIntent] = None
        for utt in utterances:
            try:
                m = engine.calc_intent(utt)
            except Exception as e:
                LOG.error(f"DomainLinhaFina calc_intent error: {e}")
                continue
            if m is None or m.name is None:
                continue
            # INTENT-4 §8.5 — disabled intents are excluded from match candidacy
            if m.name in self.disabled_intents:
                continue
            if best is None or m.conf > best.conf:
                best = LinhaFinaIntent(sent=utt, name=m.name,
                                       conf=m.conf, matches=m.slots)
        return best


@lru_cache(maxsize=3)  # repeat calls under different conf levels wont re-run code
def _calc_lf_intent(utt: str, intent_container: IntentEngine,
                    blacklisted_intents: frozenset = frozenset(),
                    blacklisted_skills: frozenset = frozenset()) -> Optional[LinhaFinaIntent]:
    """
    Try to match an utterance to an intent in an intent_container, respecting
    the session blacklists.

    The session blacklists are passed as hashable frozensets so this stays
    ``lru_cache``-able (Session is unhashable under ovos-bus-client>=2.5.1a1).

    @return: matched LinhaFinaIntent
    """
    try:
        intents = [i for i in intent_container.predict(utt)
                   if i is not None
                   and i.conf >= 0.2
                   and i.name not in blacklisted_intents
                   and i.name.split(":")[0] not in blacklisted_skills]
        LOG.debug(f"LinhaFina Intents: {intents}")
        if len(intents) == 0:
            return None
        best_conf = max(x.conf for x in intents)
        ties = [i for i in intents if i.conf == best_conf]

        # TODO - how to disambiguate ?
        best_intent = ties[0]

        intent = LinhaFinaIntent(sent=utt,
                                 name=best_intent.name,
                                 conf=best_intent.conf,
                                 matches=best_intent.slots)
        return intent
    except Exception as e:
        LOG.error(e)
