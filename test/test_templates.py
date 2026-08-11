"""Tests for linha_fina.templates."""

import pytest

from linha_fina.templates import TemplateMatcher, expand_template, expand_slots


class TestExpandTemplate:
    def test_no_expansion_returns_input(self):
        result = expand_template("hello world")
        assert result == ["hello world"]

    def test_optional_brackets(self):
        result = expand_template("sentences have [optional] words")
        assert "sentences have  words" in result or "sentences have words" in result
        assert "sentences have optional words" in result

    def test_alternatives(self):
        result = expand_template("alternative words can be (used|written)")
        assert "alternative words can be used" in result
        assert "alternative words can be written" in result

    def test_optional_plus_alternative(self):
        result = expand_template("[hello,] (call me|my name is) {name}")
        assert "call me {name}" in result
        assert "my name is {name}" in result
        assert "hello, call me {name}" in result
        assert "hello, my name is {name}" in result

    def test_nested_alternatives(self):
        result = expand_template("do( the | )thing(s|) (old|with) style")
        # 2 * 2 * 2 = 8 permutations
        assert len(result) == 8
        assert "do the things old style" in result
        assert "do thing with style" in result

    def test_inline_alternative_inside_word(self):
        result = expand_template("sentence[s] can have (pre|suf)fixes mid word too")
        assert "sentences can have prefixes mid word too" in result
        assert "sentence can have suffixes mid word too" in result

    def test_optional_whole_clause(self):
        # Per OVOS-INTENT-1 §3.6, templates yielding an empty sample are
        # malformed. The legacy implementation silently produced an empty
        # string; ovos_spec_tools.expand rejects it.
        import pytest
        from ovos_spec_tools import MalformedTemplate
        with pytest.raises(MalformedTemplate):
            expand_template("[(this|that) is optional]")

    def test_slot_placeholders_preserved(self):
        result = expand_template("tell me a [{joke_type}] joke")
        assert "tell me a {joke_type} joke" in result
        assert any("tell me a" in s and "joke" in s and "{joke_type}" not in s for s in result)


class TestExpandSlots:
    def test_single_slot_single_value(self):
        result = expand_slots("play {song}", {"song": ["africa"]})
        assert result == ["play africa"]

    def test_single_slot_multi_values(self):
        result = expand_slots("play {song}", {"song": ["africa", "hey jude"]})
        assert set(result) == {"play africa", "play hey jude"}

    def test_multi_slot_cartesian_product(self):
        result = expand_slots(
            "set {color} at {level}",
            {"color": ["red", "blue"], "level": ["low", "high"]},
        )
        assert len(result) == 4
        assert "set red at low" in result
        assert "set blue at high" in result

    def test_optional_plus_slots(self):
        result = expand_slots(
            "change [the ]color to {color}",
            {"color": ["red", "green"]},
        )
        assert "change the color to red" in result
        assert any("change color to green" in s or "change  color to green" in s for s in result)

    def test_unknown_slot_preserved(self):
        # Slot not in dict — placeholder stays
        result = expand_slots("play {song}", {})
        assert result == ["play {song}"]

    def test_no_slots_in_template(self):
        result = expand_slots("just text", {"foo": ["bar"]})
        assert result == ["just text"]


class TestTemplateMatcher:
    def test_add_templates_no_slot_is_ignored(self):
        tm = TemplateMatcher()
        tm.add_templates(["plain text with no slots"])
        # Templates without {slot} are skipped entirely
        assert all(not v for v in tm.templates.values())

    def test_simple_match(self):
        tm = TemplateMatcher()
        tm.add_templates(["play {song}"])
        result = tm.match("play africa")
        assert len(result) >= 1
        assert result[0] == {"song": "africa"}

    def test_no_match_returns_empty(self):
        tm = TemplateMatcher()
        tm.add_templates(["play {song}"])
        result = tm.match("what's the weather")
        assert result == []

    def test_multiple_templates_best_first(self):
        tm = TemplateMatcher()
        tm.add_templates(["play {song}", "put on {song}"])
        result = tm.match("put on africa")
        assert len(result) >= 1
        assert result[0] == {"song": "africa"}

    def test_two_slot_extraction(self):
        tm = TemplateMatcher()
        tm.add_templates(["set {color} to {level}"])
        result = tm.match("set red to high")
        assert result and result[0] == {"color": "red", "level": "high"}
