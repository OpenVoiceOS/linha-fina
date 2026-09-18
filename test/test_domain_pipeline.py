"""E2E tests for the DomainLinhaFinaPipeline."""

import pytest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from linha_fina.opm import DomainLinhaFinaPipeline, _split_intent_label


def _make_register_msg(name: str, samples, lang="en-US"):
    return Message("padatious:register_intent",
                   {"name": name, "samples": samples, "lang": lang})


@pytest.fixture
def pipe():
    p = DomainLinhaFinaPipeline(bus=FakeBus(), config={})
    yield p
    p.shutdown()


def test_split_intent_label():
    assert _split_intent_label("media.skill:play") == ("media.skill", "play")
    assert _split_intent_label("noskill") == ("noskill", "noskill")


def test_parallel_argmax_routes_to_correct_domain(pipe):
    # Parallel-argmax: every per-domain engine scores the utterance and the
    # highest-confidence match across domains wins. The intents in each
    # domain are deliberately disjoint in vocabulary so the routing is
    # unambiguous on the test utterances.
    pipe.register_intent(_make_register_msg(
        "media.skill:play",
        ["play music", "put on a song", "play africa", "i want to hear something",
         "start the playlist", "play some tunes"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:stop",
        ["stop the music", "halt playback", "pause the song", "stop the playlist",
         "silence the tunes", "end playback"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:next",
        ["next track", "skip song", "next track please", "skip this song",
         "go to the next track", "skip to the next song"]))
    pipe.register_intent(_make_register_msg(
        "weather.skill:forecast",
        ["what is the weather", "weather forecast", "tell me the weather",
         "is it raining", "weather report", "today's weather forecast"]))
    pipe.register_intent(_make_register_msg(
        "weather.skill:temperature",
        ["what is the temperature", "how hot is it", "current temperature",
         "how cold is it", "tell me the temperature", "what's the outdoor temp"]))
    pipe.register_intent(_make_register_msg(
        "weather.skill:rain",
        ["will it rain", "rain forecast", "is rain expected",
         "chance of rain", "rain probability", "is it going to rain today"]))

    # In-domain media utterance → media.skill
    match = pipe.calc_intent(["play music"], lang="en-US")
    assert match is not None
    skill_id, _ = _split_intent_label(match.name)
    assert skill_id == "media.skill"

    # In-domain weather utterance → weather.skill
    match2 = pipe.calc_intent(["what is the weather"], lang="en-US")
    assert match2 is not None
    skill_id2, _ = _split_intent_label(match2.name)
    assert skill_id2 == "weather.skill"


def test_handle_initial_train_pretrains_domain_engines(pipe):
    # Regression: `mycroft.ready` used to call `.train()` on every per-lang
    # container, but DomainIntentEngine (unlike the flat IntentEngine) had no
    # `.train()` method of its own -> AttributeError, so pre-training silently
    # never happened and every subsequent match failed until lazy
    # train-on-first-prediction kicked in inside the per-domain engines.
    pipe.register_intent(_make_register_msg(
        "media.skill:play",
        ["play music", "put on a song", "play africa", "start the playlist"],
        lang=pipe.lang))
    pipe.register_intent(_make_register_msg(
        "media.skill:stop",
        ["stop the music", "halt playback", "pause the song", "stop the playlist"],
        lang=pipe.lang))

    engine = pipe.containers[pipe.lang]
    domain_engine = engine.domains["media.skill"]
    assert domain_engine.clf._needs_training is True

    # Must not raise AttributeError, and must actually train the sub-engines.
    pipe.handle_initial_train(Message("mycroft.ready", {}))

    assert domain_engine.clf._needs_training is False


def test_detach_skill_removes_domain(pipe):
    for name, samples in [
        ("media.skill:play", ["play music", "play a song", "put on tunes", "play africa"]),
        ("media.skill:stop", ["stop", "halt", "pause it", "silence"]),
        ("media.skill:next", ["next", "skip", "next track", "skip this"]),
    ]:
        pipe.register_intent(_make_register_msg(name, samples))

    engine = pipe.containers["en-US"]
    assert "media.skill" in engine.domains

    pipe.handle_detach_skill(Message("detach_skill", {"skill_id": "media.skill"}))
    assert "media.skill" not in engine.domains
    assert not any(i.startswith("media.skill:") for i in pipe.registered_intents)


def test_ovoscope_harness_loads_domain_plugin():
    """Verify the domain pipeline OPM entry point resolves inside MiniCroft."""
    pytest.importorskip("ovoscope")
    from ovoscope.pipeline import PipelineHarness

    plugin_id = "ovos-linha-fina-domain-pipeline-plugin"
    with PipelineHarness(pipeline=[plugin_id], lang="en-US") as h:
        loaded = h._mc.intents.pipeline_plugins
        assert plugin_id in loaded
        assert isinstance(loaded[plugin_id], DomainLinhaFinaPipeline)
        # Asserting the *live plugin* received and registered an intent via the
        # real OVOS bus — not via a side-channel listener.
        h._mc.bus.emit(_make_register_msg(
            "media.skill:play",
            ["play music", "put on a song", "play africa"]))
        plugin = loaded[plugin_id]
        assert "media.skill:play" in plugin.registered_intents
        assert "media.skill" in plugin.containers[plugin.lang].domains
