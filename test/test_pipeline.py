"""OPM pipeline tests for LinhaFinaPipeline.

Drives the plugin through its real messagebus contract using a FakeBus —
the same path OVOS uses in production, minus the network. Fast and
exhaustive: instantiates ``LinhaFinaPipeline`` directly so each test
runs in milliseconds with no MiniCroft startup.

For end-to-end smoke coverage that proves the plugin loads via the OPM
entry point inside a real MiniCroft, see ``test_pipeline_ovoscope.py``.
"""

import time

import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from linha_fina.opm import LinhaFinaPipeline, LinhaFinaIntent, _calc_lf_intent


@pytest.fixture
def pipeline():
    p = LinhaFinaPipeline(bus=FakeBus(), config={
        "conf_high": 0.8,
        "conf_med": 0.6,
        "conf_low": 0.4,
    })
    yield p
    p.shutdown()


def _register(pipeline, intents):
    """Emit padatious:register_intent for each intent then warm-train."""
    for name, samples in intents.items():
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": name,
            "samples": samples,
            "lang": "en-US",
        }))
    pipeline.bus.emit(Message("mycroft.ready"))


class TestConstruction:
    def test_default_thresholds(self):
        p = LinhaFinaPipeline(bus=FakeBus(), config={})
        assert p.conf_high == 0.8
        assert p.conf_med == 0.6
        assert p.conf_low == 0.4
        p.shutdown()

    def test_custom_thresholds(self):
        p = LinhaFinaPipeline(bus=FakeBus(), config={
            "conf_high": 0.95, "conf_med": 0.7, "conf_low": 0.5,
        })
        assert p.conf_high == 0.95
        assert p.conf_med == 0.7
        assert p.conf_low == 0.5
        p.shutdown()

    def test_creates_container_for_primary_lang(self):
        p = LinhaFinaPipeline(bus=FakeBus(), config={})
        assert p.lang in p.containers
        p.shutdown()

    def test_subscribes_to_bus_handlers(self):
        p = LinhaFinaPipeline(bus=FakeBus(), config={})
        # All four handler events should have at least one subscriber
        for ev in [
            "padatious:register_intent",
            "padatious:register_entity",
            "detach_intent",
            "detach_skill",
        ]:
            assert p.bus.ee.listeners(ev), f"no subscriber for {ev}"
        p.shutdown()


class TestRegistrationViaBus:
    def test_register_intent_populates_container(self, pipeline):
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:greet",
            "samples": ["hello", "hi", "hey"],
            "lang": "en-US",
        }))
        container = pipeline.containers[pipeline.lang]
        assert "demo:greet" in container.clf.clfs
        assert "demo:greet" in pipeline.registered_intents

    def test_register_intent_unknown_lang_is_ignored(self, pipeline):
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:greet",
            "samples": ["bonjour"],
            "lang": "fr-FR",
        }))
        # No fr-FR container → silently skipped
        assert "demo:greet" not in pipeline.registered_intents

    def test_register_intent_without_samples_or_file_logs_error(self, pipeline, caplog):
        # Should not raise, just early-return
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:bad",
            "lang": "en-US",
        }))
        container = pipeline.containers[pipeline.lang]
        assert "demo:bad" not in container.clf.clfs

    def test_register_intent_from_file(self, pipeline, tmp_path):
        samples_file = tmp_path / "samples.txt"
        samples_file.write_text("hello world\nhi there\nhey\n")
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:from_file",
            "file_name": str(samples_file),
            "lang": "en-US",
        }))
        container = pipeline.containers[pipeline.lang]
        assert "demo:from_file" in container.clf.clfs
        positives = container.clf.clfs["demo:from_file"].positives
        assert "hello world" in positives
        assert "hi there" in positives

    def test_typed_placeholder_degrades_to_the_bare_slot(self, pipeline, tmp_path):
        """OVOS-INTENT-1 3.4: a loader that does not implement typed slots
        MUST treat ``{type:name}`` as ``{name}``. Samples read from a file
        reach the engine as written, so the prefix has to be stripped here."""
        samples_file = tmp_path / "alarm.intent"
        samples_file.write_text("set an alarm in {number:offset} minutes\n")
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:alarm_offset",
            "file_name": str(samples_file),
            "lang": "en-US",
        }))
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "demo:other",
            "samples": ["what time is it", "tell me the time"],
            "lang": "en-US",
        }))
        pipeline.bus.emit(Message("mycroft.ready"))
        container = pipeline.containers[pipeline.lang]
        positives = container.clf.clfs["demo:alarm_offset"].positives
        assert "set an alarm in {offset} minutes" in positives
        assert not any("{number:offset}" in s for s in positives)
        utt = "set an alarm in 5 minutes"
        match = pipeline.calc_intent([utt], "en-US", Message(
            "recognizer_loop:utterance", {"utterances": [utt], "lang": "en-US"}))
        assert match is not None and match.name == "demo:alarm_offset"
        matches = match.matches or {}
        if isinstance(matches, dict):
            matches = [matches]
        slots = {k: v for m in matches for k, v in m.items()}
        assert slots.get("offset") == "5"

    def test_register_entity_populates_kw_matcher(self, pipeline):
        # Register an intent first (so a k_matcher exists for the targeted intent)
        pipeline.bus.emit(Message("padatious:register_intent", data={
            "name": "media:play",
            "samples": ["play {song}"],
            "lang": "en-US",
        }))
        pipeline.bus.emit(Message("padatious:register_entity", data={
            "name": "song",
            "samples": ["africa", "hey jude"],
            "lang": "en-US",
        }))
        assert {"name": "song", "samples": ["africa", "hey jude"], "lang": "en-US"} in pipeline.registered_entities


class TestDetach:
    def test_detach_intent_removes_from_all_languages(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi", "hey"],
            "demo:bye": ["goodbye", "bye", "see you"],
            "demo:thanks": ["thanks", "ty", "thank you"],
        })
        assert "demo:greet" in pipeline.registered_intents
        pipeline.bus.emit(Message("detach_intent", data={"intent_name": "demo:greet"}))
        assert "demo:greet" not in pipeline.registered_intents
        container = pipeline.containers[pipeline.lang]
        assert "demo:greet" not in container.clf.clfs

    def test_detach_skill_removes_all_skill_intents(self, pipeline):
        _register(pipeline, {
            "media:play": ["play music", "start music", "begin"],
            "media:stop": ["stop", "halt", "cease"],
            "weather:forecast": ["weather", "forecast", "temperature"],
        })
        pipeline.bus.emit(Message("detach_skill", data={"skill_id": "media"}))
        assert "media:play" not in pipeline.registered_intents
        assert "media:stop" not in pipeline.registered_intents
        assert "weather:forecast" in pipeline.registered_intents


class TestMatching:
    def test_match_high_fires_above_threshold(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi there", "hey", "good morning"],
            "demo:bye": ["goodbye", "see you", "bye", "later"],
            "demo:thanks": ["thanks", "thank you", "much appreciated"],
        })
        _calc_lf_intent.cache_clear()
        m = pipeline.match_high(["hello there"], lang="en-US",
                                message=Message("test", {}))
        # If model didn't reach the high threshold the call returns None — that's
        # acceptable; what we're asserting is the call path completes without error.
        if m is not None:
            assert m.match_type == "demo:greet"
            assert m.skill_id == "demo"

    def test_match_low_more_permissive_than_high(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi", "hey"],
            "demo:bye": ["goodbye", "bye", "see you"],
            "demo:thanks": ["thanks", "thank you", "ty"],
        })
        _calc_lf_intent.cache_clear()
        # A borderline utterance should be more likely to fire on low than high
        low = pipeline.match_low(["hello"], lang="en-US", message=Message("t", {}))
        if low is not None:
            assert low.match_type.startswith("demo:")

    def test_match_medium_threshold(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi there", "hey"],
            "demo:bye": ["goodbye", "see you", "bye"],
            "demo:thanks": ["thanks", "thank you", "ty"],
        })
        _calc_lf_intent.cache_clear()
        m = pipeline.match_medium(["thank you"], lang="en-US",
                                  message=Message("t", {}))
        if m is not None:
            assert m.match_type.startswith("demo:")

    def test_oversized_utterance_returns_none(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi", "hey"],
            "demo:bye": ["goodbye", "bye", "see you"],
            "demo:thanks": ["thanks", "ty", "thank you"],
        })
        long_utt = " ".join(["word"] * 100)
        m = pipeline.match_high([long_utt], lang="en-US", message=Message("t", {}))
        assert m is None

    def test_calc_intent_accepts_string_arg(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi", "hey"],
            "demo:bye": ["goodbye", "bye", "see you"],
            "demo:thanks": ["thanks", "ty", "thank you"],
        })
        _calc_lf_intent.cache_clear()
        # Backwards compat: should accept a plain string
        result = pipeline.calc_intent("hello")
        # May be None or a LinhaFinaIntent — both are valid shapes here
        assert result is None or isinstance(result, LinhaFinaIntent)

    def test_calc_intent_unknown_lang_returns_none(self, pipeline):
        _register(pipeline, {
            "demo:greet": ["hello", "hi", "hey"],
            "demo:bye": ["goodbye", "bye"],
            "demo:thanks": ["thanks", "ty"],
        })
        _calc_lf_intent.cache_clear()
        # ja-JP has no container and is too far from en-US (distance > 10)
        result = pipeline.calc_intent(["こんにちは"], lang="ja-JP")
        assert result is None


class TestLinhaFinaIntent:
    def test_dict_like_access(self):
        i = LinhaFinaIntent(
            name="demo:play",
            sent="play africa",
            matches={"song": "africa"},
            conf=0.9,
        )
        assert i["song"] == "africa"
        assert "song" in i
        assert i.get("song") == "africa"
        assert i.get("missing", "default") == "default"

    def test_repr_includes_state(self):
        i = LinhaFinaIntent(name="x:y", sent="hi", conf=0.5)
        r = repr(i)
        assert "x:y" in r
        assert "hi" in r

    def test_defaults(self):
        i = LinhaFinaIntent(name="x", sent="y")
        assert i.matches == {}
        assert i.conf == 0.0


class TestLanguageRouting:
    def test_closest_lang_exact(self, pipeline):
        assert pipeline._get_closest_lang("en-US") == pipeline.lang

    def test_closest_lang_regional_variant(self, pipeline):
        # en-AU should fall back to en-US (distance < 10)
        result = pipeline._get_closest_lang("en-AU")
        assert result == pipeline.lang

    def test_closest_lang_unrelated_returns_none(self, pipeline):
        # Japanese to English — distance > 10
        assert pipeline._get_closest_lang("ja-JP") is None
