"""T-3165: ``IntentHandlerMatch.match_data`` must be the ``{str: str}`` slot
map of OVOS-PIPELINE-1 §4.1/§4.3. ``TemplateMatcher.match`` returns every
candidate slot dict, ranked, and ``IntentEngine.predict`` stored that list
as the match's slots. ovos-core's ``_dispatch_match`` does
``data.update(match.match_data)`` and raised ``ValueError: dictionary update
sequence element #0 has length 1`` on every template-matched intent."""
import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from linha_fina.engine import IntentEngine
from linha_fina.opm import LinhaFinaPipeline

SAMPLES = ["set the brightness to {b}", "brightness {b}"]


def test_engine_slots_are_a_flat_mapping():
    engine = IntentEngine(instant_train=True)
    engine.register_intent("demo:set_brightness", SAMPLES)
    engine.register_intent("demo:volume", ["set the volume to {v}", "volume {v}"])
    engine.train()
    match = engine.calc_intent("set the brightness to twenty five")
    assert match.name == "demo:set_brightness"
    assert isinstance(match.slots, dict), match.slots
    assert match.slots == {"b": "twenty five"}


def test_match_data_survives_core_dispatch():
    """Fail-before: the real ovos-core dispatch path, not a reimplementation."""
    from ovos_core.intent_services.service import IntentService

    bus = FakeBus()
    plugin = LinhaFinaPipeline(bus=bus, config={"conf_high": 0.8, "conf_med": 0.6, "conf_low": 0.4})
    try:
        for name, samples in {"demo:set_brightness": SAMPLES,
                              "demo:volume": ["set the volume to {v}", "volume {v}"]}.items():
            bus.emit(Message("padatious:register_intent",
                             data={"name": name, "samples": samples, "lang": "en-US"}))
        bus.emit(Message("mycroft.ready"))
        utt = "set the brightness to twenty five"
        message = Message("recognizer_loop:utterance", {"utterances": [utt], "lang": "en-US"})
        match = plugin.match_high([utt], "en-US", message)
        assert match is not None and match.match_type == "demo:set_brightness"

        service = IntentService(bus, preload_pipelines=False)
        dispatched = []
        bus.on("demo:set_brightness", dispatched.append)
        service._dispatch_match(match, message, "en-US", "ovos-linha-fina-pipeline-plugin")
        assert dispatched, "no dispatch Message reached the handler topic"
        assert dispatched[0].data["b"] == "twenty five"
    finally:
        plugin.shutdown()
