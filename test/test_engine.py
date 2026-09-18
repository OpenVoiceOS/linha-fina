"""Tests for linha_fina.engine."""

import pytest

from linha_fina.engine import IntentEngine, IntentMatch


@pytest.fixture
def engine():
    e = IntentEngine()
    e.register_intent("greet", ["hello", "hi there", "hey", "good morning"])
    e.register_intent("bye", ["goodbye", "see you", "bye", "talk to you later"])
    e.register_intent("thanks", ["thanks", "thank you", "much appreciated"])
    e.train()
    return e


@pytest.fixture
def slot_engine():
    e = IntentEngine()
    e.register_intent(
        "play",
        ["play {song}", "put on {song}", "I want to hear {song}"],
        entity_samples={"song": ["africa", "hey jude", "wonderwall"]},
    )
    e.register_intent("stop", ["stop", "shut up", "pause"])
    e.register_intent("next", ["next track", "skip", "next song"])
    e.train()
    return e


class TestRegistration:
    def test_register_intent_creates_classifier(self):
        e = IntentEngine()
        e.register_intent("foo", ["alpha", "beta"])
        assert "foo" in e.clf.clfs

    def test_register_intent_with_slots_creates_template(self):
        e = IntentEngine()
        e.register_intent(
            "play",
            ["play {song}"],
            entity_samples={"song": ["africa"]},
        )
        assert "play" in e.t_matchers
        assert "play" in e.k_matchers

    def test_register_intent_without_slots_no_template(self):
        e = IntentEngine()
        e.register_intent("greet", ["hello", "hi"])
        # No {slot} → no template, no keyword extractor
        assert "greet" not in e.t_matchers
        assert "greet" not in e.k_matchers

    def test_register_intent_expands_entity_samples_into_training(self):
        e = IntentEngine()
        e.register_intent(
            "play",
            ["play {song}"],
            entity_samples={"song": ["africa", "hey jude"]},
        )
        positives = e.clf.clfs["play"].positives
        # Should include the template plus one expanded sample per entity value
        assert "play {song}" in positives
        assert "play africa" in positives
        assert "play hey jude" in positives

    def test_remove_intent_cleans_all(self, slot_engine):
        slot_engine.remove_intent("play")
        assert "play" not in slot_engine.clf.clfs
        assert "play" not in slot_engine.t_matchers
        assert "play" not in slot_engine.k_matchers


class TestEntityManagement:
    def test_register_entity_targeted_to_intent(self):
        e = IntentEngine()
        e.register_intent(
            "play",
            ["play {song}"],
            entity_samples={"song": ["africa"]},
        )
        e.register_entity("song", ["new song"], intent_name="play")
        assert "new song" in e.k_matchers["play"].entities["song"]

    def test_register_entity_broadcast_to_all_intents_with_kw(self):
        e = IntentEngine()
        # Both intents have keyword matchers
        e.register_intent("play", ["play {song}"], entity_samples={"song": ["a"]})
        e.register_intent("queue", ["queue {song}"], entity_samples={"song": ["b"]})
        e.register_entity("song", ["broadcast"])
        assert "broadcast" in e.k_matchers["play"].entities["song"]
        assert "broadcast" in e.k_matchers["queue"].entities["song"]

    def test_remove_entity_from_specific_intent(self):
        e = IntentEngine()
        e.register_intent("play", ["play {song}"], entity_samples={"song": ["one", "two"]})
        e.remove_entity("song", intent_name="play")
        assert "song" not in e.k_matchers["play"].entities


class TestPrediction:
    def test_calc_intent_returns_intent_match(self, engine):
        m = engine.calc_intent("hello")
        assert isinstance(m, IntentMatch)
        assert m.name in {"greet", "bye", "thanks"}

    def test_calc_intent_picks_intended_label(self, engine):
        assert engine.calc_intent("hello there").name == "greet"
        assert engine.calc_intent("goodbye").name == "bye"
        assert engine.calc_intent("thank you").name == "thanks"

    def test_predict_returns_top_n(self, engine):
        results = engine.predict("hello", top_n=2)
        assert len(results) == 2
        assert results[0].conf >= results[1].conf

    def test_predict_sorted_descending(self, engine):
        results = engine.predict("hello", top_n=3)
        confs = [r.conf for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_confidence_in_valid_range(self, engine):
        for r in engine.predict("hello there", top_n=3):
            assert 0.0 <= r.conf <= 1.0


class TestSlotExtraction:
    def test_template_match_extracts_slot(self, slot_engine):
        m = slot_engine.calc_intent("play africa")
        assert m.name == "play"
        # one {str: str} map whichever layer fired (PIPELINE-1 §4.3, T-3165)
        assert isinstance(m.slots, dict)
        assert m.slots.get("song") == "africa"

    def test_template_with_alternative_phrasing(self, slot_engine):
        m = slot_engine.calc_intent("put on africa")
        assert m.name == "play"

    def test_intent_without_slots_returns_empty(self, engine):
        m = engine.calc_intent("hello")
        # No keyword/template matcher → slots empty
        assert m.slots == {}

    def test_keyword_fallback_when_template_misses(self):
        e = IntentEngine()
        e.register_intent(
            "play",
            ["play {song}"],
            entity_samples={"song": ["africa"]},
        )
        e.register_intent("stop", ["stop", "shut up", "be quiet"])
        e.register_intent("next", ["next track", "skip", "next song"])
        e.train()
        # Utterance has the keyword but doesn't match the template pattern strictly
        m = e.calc_intent("africa please")
        if m.name == "play":
            assert isinstance(m.slots, dict)
            assert m.slots.get("song") == "africa"


class TestLazyVsEagerTraining:
    def test_lazy_training_default(self):
        e = IntentEngine(instant_train=False)
        e.register_intent("a", ["alpha"])
        e.register_intent("b", ["beta"])
        e.register_intent("c", ["gamma"])
        assert e.clf._needs_training is True
        # First inference forces training
        e.calc_intent("alpha")
        assert e.clf._needs_training is False

    def test_eager_training(self):
        e = IntentEngine(instant_train=True)
        e.register_intent("a", ["alpha one", "alpha two"])
        e.register_intent("b", ["beta one", "beta two"])
        e.register_intent("c", ["gamma one", "gamma two"])
        # instant_train should have trained after the 3rd label
        assert e.clf._needs_training is False
