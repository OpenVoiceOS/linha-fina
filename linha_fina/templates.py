import itertools
import re
import warnings
from collections import defaultdict
from typing import List, Dict

from ovos_spec_tools import expand
from ovos_utils.log import deprecated
from rapidfuzz import fuzz
from simplematch import match as sm

from linha_fina.version import VERSION_MAJOR


class TemplateMatcher:
    """
    Matches text to predefined templates using slot filling and fuzzy matching.
    """

    def __init__(self):
        self.templates: Dict[str, List[str]] = defaultdict(list)

    def add_templates(self, templates: List[str]) -> None:
        """
        Adds templates for matching, extracting slots from templates.

        Args:
            templates (List[str]): A list of templates with slots.
        """
        for template in templates:
            # Extract words within {curly_braces} using regex
            slots = re.findall(r"\{(\w+)\}", template)
            if not slots:
                continue
            t_name = slots[0]
            self.templates[t_name].append(template)

    def match(self, query: str) -> List[Dict[str, str]]:
        """
        Matches the input query to a template.

        Args:
            query (str): The input query.

        Returns:
            Slots: A dictionary with matched slots and confidence score.
        """
        matches = []
        for ent, templates in self.templates.items():
            for t in templates:
                m = sm(t, query)
                if m:
                    s = fuzz.token_set_ratio(t, query)
                    matches.append((s, m))
        matches = sorted(matches, key=lambda k: k[0], reverse=True)
        return [m[1] for m in matches]


@deprecated("Use ovos_spec_tools.expand instead", f"{VERSION_MAJOR + 1}.0.0")
def expand_template(template: str) -> list[str]:
    """Deprecated shim — use :func:`ovos_spec_tools.expand`."""
    warnings.warn(
        "linha_fina.templates.expand_template is deprecated; "
        "use ovos_spec_tools.expand instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sorted(set(expand(template)))


def expand_slots(template: str, slots: dict[str, list[str]]) -> list[str]:
    """Expand a template by first expanding alternatives and optional components,
    then substituting slot placeholders with their corresponding options.

    Args:
        template (str): The input string template to expand.
        slots (dict): A dictionary where keys are slot names and values are lists of possible replacements.

    Returns:
        list[str]: A list of all expanded combinations.
    """
    # Expand alternatives and optional components
    base_expansions = expand(template)

    # Process slots
    all_sentences = []
    for sentence in base_expansions:
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if matches:
            # Create all combinations for slots in the sentence
            slot_options = [slots.get(match, [f"{{{match}}}"]) for match in matches]
            for combination in itertools.product(*slot_options):
                filled_sentence = sentence
                for slot, replacement in zip(matches, combination):
                    filled_sentence = filled_sentence.replace(f"{{{slot}}}", replacement)
                all_sentences.append(filled_sentence)
        else:
            # No slots to expand
            all_sentences.append(sentence)

    return all_sentences


if __name__ == "__main__":

    template = "change [the ]brightness to {brightness_level} and color to {color_name}"
    slots = {
        "brightness_level": ["low", "medium", "high"],
        "color_name": ["red", "green", "blue"]
    }

    expanded_sentences = expand_slots(template, slots)
    for sentence in expanded_sentences:
        print(sentence)
        # change the brightness to low and color to red
        # change the brightness to low and color to green
        # change the brightness to low and color to blue
        # change the brightness to medium and color to red
        # change the brightness to medium and color to green
        # change the brightness to medium and color to blue
        # change the brightness to high and color to red
        # change the brightness to high and color to green
        # change the brightness to high and color to blue
        # change brightness to low and color to red
        # change brightness to low and color to green
        # change brightness to low and color to blue
        # change brightness to medium and color to red
        # change brightness to medium and color to green
        # change brightness to medium and color to blue
        # change brightness to high and color to red
        # change brightness to high and color to green
        # change brightness to high and color to blue

    templates = [
        "[hello,] (call me|my name is) {name}",
        "Expand (alternative|choices) into a list of choices.",
        "sentences have [optional] words ",
        "alternative words can be (used|written)",
        "sentence[s] can have (pre|suf)fixes mid word too",
        "do( the | )thing(s|) (old|with) style and( no | )spaces",
        "[(this|that) is optional]",
        "tell me a [{joke_type}] joke",
        "play {query} [in ({device_name}|{skill_name}|{zone_name})]"
    ]
    for template in templates:
        print("###", template)
        expanded_sentences = expand(template)
        for sentence in expanded_sentences:
            print("-", sentence)
        # ### [hello,] (call me|my name is) {name}
        # - call me {name}
        # - hello, call me {name}
        # - hello, my name is {name}
        # - my name is {name}
        # ### Expand (alternative|choices) into a list of choices.
        # - Expand alternative into a list of choices.
        # - Expand choices into a list of choices.
        # ### sentences have [optional] words
        # - sentences have  words
        # - sentences have optional words
        # ### alternative words can be (used|written)
        # - alternative words can be used
        # - alternative words can be written
        # ### sentence[s] can have (pre|suf)fixes mid word too
        # - sentence can have prefixes mid word too
        # - sentence can have suffixes mid word too
        # - sentences can have prefixes mid word too
        # - sentences can have suffixes mid word too
        # ### do( the | )thing(s|) (old|with) style and( no | )spaces
        # - do the thing old style and no spaces
        # - do the thing old style and spaces
        # - do the thing with style and no spaces
        # - do the thing with style and spaces
        # - do the things old style and no spaces
        # - do the things old style and spaces
        # - do the things with style and no spaces
        # - do the things with style and spaces
        # - do thing old style and no spaces
        # - do thing old style and spaces
        # - do thing with style and no spaces
        # - do thing with style and spaces
        # - do things old style and no spaces
        # - do things old style and spaces
        # - do things with style and no spaces
        # - do things with style and spaces
        # ### [(this|that) is optional]
        # -
        # - that is optional
        # - this is optional
        # ### tell me a [{joke_type}] joke
        # - tell me a  joke
        # - tell me a {joke_type} joke
        # ### play {query} [in ({device_name}|{skill_name}|{zone_name})]
        # - play {query}
        # - play {query} in {device_name}
        # - play {query} in {skill_name}
        # - play {query} in {zone_name}
