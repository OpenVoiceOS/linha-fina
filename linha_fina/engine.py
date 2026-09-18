import dataclasses
from collections import defaultdict
from typing import List, Optional, Dict

from linha_fina.dynamic import DynamicClassifier
from linha_fina.keywords import KeywordFeatures
from linha_fina.templates import TemplateMatcher


@dataclasses.dataclass
class IntentMatch:
    name: str
    slots: Dict[str, str]
    conf: float


class IntentEngine:
    def __init__(self, instant_train=False):
        self.clf = DynamicClassifier(instant_train=instant_train)
        self.t_matchers: Dict[str, TemplateMatcher] = defaultdict(TemplateMatcher)
        self.k_matchers: Dict[str, KeywordFeatures] = defaultdict(KeywordFeatures)

    def train(self):
        self.clf.train()

    def register_intent(self, name: str,
                        samples: List[str],
                        entity_samples: Optional[Dict[str, List[str]]] = None):
        samples = samples or []
        templates = [s for s in samples if "{" in s and "}" in s]
        entity_samples = entity_samples or {}
        extra_samples = []  # generated from entity + template combos
        if templates:
            self.t_matchers[name].add_templates(templates)
            for ent, e_samples in entity_samples.items():
                k = "{" + ent + "}"
                extra_samples += [t.replace(k, s)
                                  for t in templates for s in e_samples
                                  if k in t]

        self.clf.add_label(name, samples + extra_samples)

        if entity_samples:
            for ent, e_samples in entity_samples.items():
                self.k_matchers[name].register_entity(ent, e_samples)

    def remove_intent(self, name: str):
        self.clf.remove_label(name)

        if name in self.k_matchers:
            self.k_matchers.pop(name)

        if name in self.t_matchers:
            self.t_matchers.pop(name)

    def register_entity(self, name: str, samples: List[str],
                        intent_name: Optional[str] = None):
        intents = [intent_name] if intent_name else list(self.k_matchers.keys())
        for intent in intents:
            self.k_matchers[intent].register_entity(name, samples)

    def remove_entity(self, name: str, intent_name: Optional[str] = None):
        intents = [intent_name] if intent_name else list(self.k_matchers.keys())
        for intent in [i for i in intents if i in self.k_matchers]:
            self.k_matchers[intent].deregister_entity(name)

    def calc_intent(self, query: str) -> IntentMatch:
        return self.predict(query, top_n=1)[0]

    def predict(self, query: str, top_n: int = 3) -> List[IntentMatch]:
        """
        Predict the top N intents for a query.

        Args:
            query (str): The input query.
            top_n (int): Number of top predictions to return.

        Returns:
            List[IntentMatch]: A list of top N intent matches.
        """
        preds = self.clf.predict(query)
        slots = {}
        results = []

        for label in preds:
            ents = {}
            if label in self.k_matchers:
                ents = self.k_matchers[label].extract(query)
            if label in self.t_matchers:
                if ents:
                    # increase score when keywords matched and
                    # the label has a kw extractor (even if template itself didnt match)
                    # print("benefit", label)
                    preds[label] = min(1.0, preds[label] * 1.1)

                # match() returns every candidate slot dict, best first;
                # IntentMatch.slots is the one {str: str} map PIPELINE-1
                # §4.3 dispatches, so take the best candidate (T-3165)
                candidates = self.t_matchers[label].match(query)
                ents = candidates[0] if candidates else ents
                # penalize if the template has keywords, but no keywords extracted
                if not ents:
                    # print("penalize", label)
                    preds[label] = preds[label] * 0.75

            slots[label] = ents

        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for label, conf in sorted_preds:
            results.append(IntentMatch(label, slots[label], conf))
        return results


if __name__ == "__main__":

    engine = IntentEngine()

    intent_1 = ["hello world", "hey there", "hello"]
    intent_2 = ["tell me a joke", "say a joke", "make me laugh"]
    intent_3 = ["how is the weather", "what's the weather like",
                "what is the weather outside"]
    intent_4 = ["my name is {name}", "call me {name}"]
    intent_5 = [
        "change the color to {color}",
        "change light to {color}",
        "set light to {color}}"]

    engine.register_intent("hello", intent_1)
    engine.register_intent("joke", intent_2)
    engine.register_intent("weather", intent_3)
    engine.register_intent("introduce", intent_4)
    engine.register_intent("color", intent_5,
                           entity_samples={"color": ["red", "green", "blue"]})

    for s in ["hello earth",
              "call me Casimiro",
              "my name is Miro",
              "tell me a joke",
              "what is the weather",
              "tell me a joke about the weather",
              "red blue and green are my 3 favorite colors",
              "red is my favorite color",
              "light to blue",
              "change to green",
              "make color green"
              ]:
        print(s)
        print(engine.calc_intent(s))
        # hello earth
        # IntentMatch(name='hello', slots={}, conf=0.9728585833827678)
        # call me Casimiro
        # IntentMatch(name='introduce', slots=[{'name': 'Casimiro'}], conf=0.9236423200801248)
        # my name is Miro
        # IntentMatch(name='introduce', slots=[{'name': 'Miro'}], conf=0.9949282563482034)
        # tell me a joke
        # IntentMatch(name='joke', slots={}, conf=0.9728801357230902)
        # what is the weather
        # IntentMatch(name='weather', slots={}, conf=0.9427588770219081)
        # tell me a joke about the weather
        # IntentMatch(name='joke', slots={}, conf=0.9728801357230902)
        # red blue and green are my 3 favorite colors
        # IntentMatch(name='color', slots={'color': 'green'}, conf=0.9648410361539215)
        # red is my favorite color
        # IntentMatch(name='color', slots={'color': 'red'}, conf=0.9142220645436988)
        # light to blue
        # IntentMatch(name='color', slots={'color': 'blue'}, conf=0.9845866111672085)
        # change to green
        # IntentMatch(name='color', slots={'color': 'green'}, conf=0.9598265049335444)
        # make color green
        # IntentMatch(name='color', slots={'color': 'green'}, conf=0.9142364945872282)
