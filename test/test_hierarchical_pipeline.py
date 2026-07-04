"""Tests for the HierarchicalIntentEngine and HierarchicalLinhaFinaPipeline."""

import pytest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from linha_fina.hierarchical_engine import HierarchicalIntentEngine
from linha_fina.hierarchical_opm import HierarchicalLinhaFinaPipeline, _split_intent_label


def _make_register_msg(name: str, samples, lang="en-US"):
    return Message("padatious:register_intent",
                   {"name": name, "samples": samples, "lang": lang})


@pytest.fixture
def engine():
    eng = HierarchicalIntentEngine()
    eng.register_domain_intent("media", "play",
                               ["play music", "put on a song", "play africa",
                                "i want to hear something"])
    eng.register_domain_intent("media", "stop",
                               ["stop", "stop the music", "halt playback",
                                "silence please"])
    eng.register_domain_intent("media", "next",
                               ["next track", "skip song", "next please",
                                "skip this one"])
    eng.register_domain_intent("home", "lights_on",
                               ["turn on the lights", "lights on",
                                "switch on the lamp", "illuminate the room"])
    eng.register_domain_intent("home", "lights_off",
                               ["turn off the lights", "lights off",
                                "switch off the lamp", "darken the room"])
    eng.register_domain_intent("home", "thermostat",
                               ["set temperature to twenty", "make it warmer",
                                "make it cooler", "thermostat to high"])
    return eng


@pytest.fixture
def pipe():
    p = HierarchicalLinhaFinaPipeline(bus=FakeBus(), config={})
    yield p
    p.shutdown()


def test_split_intent_label():
    assert _split_intent_label("media.skill:play") == ("media.skill", "play")
    assert _split_intent_label("noskill") == ("noskill", "noskill")


def test_calc_domain_classifies(engine):
    dom = engine.calc_domain("play some jazz")
    assert dom is not None
    assert dom.name == "media"

    dom2 = engine.calc_domain("turn on the lights")
    assert dom2 is not None
    assert dom2.name == "home"


def test_calc_intent_two_stage(engine):
    match = engine.calc_intent("play music")
    assert match is not None
    assert match.name == "play"

    match2 = engine.calc_intent("lights on")
    assert match2 is not None
    assert match2.name == "lights_on"


def test_calc_intent_explicit_domain(engine):
    # explicit domain bypasses the top-level classifier
    match = engine.calc_intent("stop", domain="media")
    assert match is not None
    assert match.name == "stop"

    # unknown explicit domain yields a no-match
    match2 = engine.calc_intent("stop", domain="nonexistent")
    assert match2.name is None


def test_domain_threshold_rejects_offtopic():
    eng = HierarchicalIntentEngine(domain_threshold=1.5)
    eng.register_domain_intent("media", "play", ["play music", "put on a song"])
    eng.register_domain_intent("media", "stop", ["stop", "stop the music"])
    eng.register_domain_intent("home", "lights_on",
                               ["lights on", "turn on the lights"])
    # threshold above 1.0 can never be reached -> always no-match
    match = eng.calc_intent("play music")
    assert match.name is None


def test_remove_domain(engine):
    # third domain keeps the top-level classifier trainable after a removal
    engine.register_domain_intent("weather", "forecast",
                                  ["what is the weather", "weather forecast",
                                   "is it going to rain", "weather today"])
    assert "media" in engine.domains
    engine.remove_domain("media")
    assert "media" not in engine.domains
    assert "media" not in engine.training_data
    dom = engine.calc_domain("play some jazz")
    assert dom.name != "media"


def test_lazy_classifier_rebuild(engine):
    # registration only marks dirty; classifier syncs on first query
    assert engine._dirty_domains
    engine.calc_domain("play music")
    assert not engine._dirty_domains


def test_pipeline_two_stage_routes_and_matches(pipe):
    pipe.register_intent(_make_register_msg(
        "media.skill:play",
        ["play music", "put on a song", "play africa", "i want to hear something"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:stop",
        ["stop", "stop the music", "halt playback", "silence please"]))
    pipe.register_intent(_make_register_msg(
        "media.skill:next",
        ["next track", "skip song", "next please", "skip this one"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:lights_on",
        ["turn on the lights", "lights on", "switch on the lamp", "illuminate the room"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:lights_off",
        ["turn off the lights", "lights off", "switch off the lamp", "darken the room"]))
    pipe.register_intent(_make_register_msg(
        "home.skill:thermostat",
        ["set temperature to twenty", "make it warmer", "make it cooler", "thermostat to high"]))

    match = pipe.calc_intent(["play music"], lang="en-US")
    assert match is not None
    skill_id, _ = _split_intent_label(match.name)
    assert skill_id == "media.skill"

    match2 = pipe.calc_intent(["turn on the lights"], lang="en-US")
    assert match2 is not None
    skill_id2, _ = _split_intent_label(match2.name)
    assert skill_id2 == "home.skill"


def test_pipeline_detach_skill_removes_domain(pipe):
    for name, samples in [
        ("media.skill:play", ["play music", "play a song", "put on tunes", "play africa"]),
        ("media.skill:stop", ["stop", "halt", "pause it", "silence"]),
        ("media.skill:next", ["next", "skip", "next track", "skip this"]),
    ]:
        pipe.register_intent(_make_register_msg(name, samples))

    engine = pipe.containers[pipe.lang]
    assert "media.skill" in engine.domains

    pipe.handle_detach_skill(Message("detach_skill", {"skill_id": "media.skill"}))
    assert "media.skill" not in engine.domains
    assert not any(i.startswith("media.skill:") for i in pipe.registered_intents)


def test_pipeline_instantiates():
    pipe = HierarchicalLinhaFinaPipeline(bus=FakeBus(), config={})
    assert pipe.lang
    assert pipe.containers
    pipe.shutdown()
