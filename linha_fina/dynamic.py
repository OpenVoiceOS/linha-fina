import random
import threading
from collections import defaultdict
from typing import List, Optional, Dict, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from quebra_frases import word_tokenize
from rapidfuzz import fuzz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron, LogisticRegressionCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from linha_fina.keywords import KeywordFeatures
from linha_fina.templates import TemplateMatcher

try:
    from ovos_utils.log import LOG
except:
    from logging import getLogger
    LOG = getLogger("LinhaFina")

# Type aliases
TrainingData = List[Tuple[List[float], str]]
Model = Union[Perceptron, SVC, LogisticRegressionCV, MLPClassifier, SGDClassifier]


class DynamicBinaryClassifier:
    """
    A binary classifier that dynamically adapts to positive and negative samples
    and uses one-hot encoding for feature extraction.
    """

    def __init__(self):
        """
        Initializes the binary classifier with empty datasets and a keyword featurizer.
        """
        self.positives: List[str] = []
        self.negatives: List[str] = []
        self.featurizer = KeywordFeatures(use_automatons=False)
        self._needs_training: bool = True
        self.model: Optional[Model] = None
        self._neg_scores: Dict[str, float] = {}

    def init_model(self, model: Optional[Model] = None) -> None:
        """
        Initializes an MLPClassifier with appropriate architecture based on featurizer labels.
        """
        self.model = model or MLPClassifier(random_state=42)  # SVC(probability=True, random_state=42)
        if not hasattr(self.model, "predict_proba"):
            self.model = CalibratedClassifierCV(self.model)

    def add_positive(self, sents: List[str]) -> None:
        """
        Adds positive samples and updates the featurizer.

        Args:
            sents (List[str]): A list of positive sentences.
        """
        self.positives += sents
        self._needs_training = True
        for s in sents:
            for tok in word_tokenize(s):
                self.featurizer.register_entity(tok, [tok])

    def add_negative(self, sents: List[str]) -> None:
        """
        Adds negative samples.

        Args:
            sents (List[str]): A list of negative sentences.
        """
        self.negatives += sents
        # don't use too much data, otherwise the classifier can just learn to never match
        max_negs = 3 * len(self.positives)
        if self.positives:
            #LOG.debug(f"Reference sample: {self.positives[0]}")
            for s in self.negatives:
                if s not in self._neg_scores:
                    self._neg_scores[s] = fuzz.token_set_ratio(s, self.positives[0])
            # select most relevant samples to keep, whatever helps disambiguate better
            if len(self.negatives) > max_negs:
                scored = sorted(self.negatives, key=lambda k: self._neg_scores[k], reverse=False)
                self.negatives = scored[:max_negs]
                #LOG.debug(f"Selected negative samples: {self.negatives}")
        self._needs_training = True

    def one_hot_encode(self, text: str) -> List[int]:
        """
        Converts text into a one-hot encoded vector using the featurizer.

        Args:
            text (str): The input text.

        Returns:
            List[int]: The one-hot encoded feature vector.
        """
        return self.featurizer.one_hot_encode(text)

    @property
    def training_data(self) -> TrainingData:
        """
        Prepares the training data by combining positive and negative samples.

        Returns:
            A list of tuples where each tuple contains a feature vector and a label.
        """
        data: TrainingData = []
        for s in self.positives:
            data.append((self.one_hot_encode(s), "intent"))
        for s in self.negatives:
            data.append((self.one_hot_encode(s), "not-intent"))
        random.shuffle(data)
        return data

    def train(self) -> None:
        """
        Trains the model using the prepared training data.
        """
        if self.positives and self.negatives:
            X, Y = zip(*self.training_data)
            if self.model is None:
                self.init_model()
            self.model.fit(list(X), list(Y))
            self._needs_training = False

    def score(self, x, y) -> float:
        """
        Evaliates the model using the prepared training data.
        """
        if self.positives and self.negatives and self.model is not None:
            return self.model.score(x, y)
        return 0.0

    def predict(self, text: str) -> float:
        """
        Predicts the probability of the input text belonging to the positive class.

        Args:
            text (str): The input text.

        Returns:
            float: The probability of the text being classified as positive.
        """
        if self._needs_training:
            self.train()
        vec = self.one_hot_encode(text)
        return self.model.predict_proba([vec])[0][0]


class DynamicClassifier:
    """
    A multi-class classifier built on multiple DynamicBinaryClassifiers.
    """

    def __init__(self, instant_train=False):
        """
        Initializes the multi-class classifier with an empty label dictionary.
        """
        self.instant_train = instant_train
        self._needs_training: bool = True
        self.clfs: Dict[str, DynamicBinaryClassifier] = defaultdict(DynamicBinaryClassifier)
        self.lock = threading.Lock()

    def add_label(self, name: str, samples: List[str]) -> None:
        """
        Adds a new label with corresponding samples.

        Args:
            name (str): The label name.
            samples (List[str]): A list of positive samples for the label.
        """
        self.clfs[name].add_positive(samples)
        if self.instant_train:
            try:
                self.train()
                return
            except:
                pass
        self._needs_training = True

    def remove_label(self, name: str):
        if name in self.clfs:
            self.clfs.pop(name)

    def train(self) -> None:
        """
        Trains all classifiers, ensuring each label has negative samples from other labels.
        """
        with self.lock:
            clfs = dict(self.clfs)  # Copy because it might change during iteration
            if len(clfs) >= 2:
                def train_single_label(name: str):
                    # Add negative samples for the classifier
                    for name2, other_clf in clfs.items():
                        if name != name2:
                            samples = other_clf.positives
                            self.clfs[name].add_negative([s for s in samples
                                                          if s not in self.clfs[name].positives
                                                          and s not in self.clfs[name].negatives])
                    # Train the classifier
                    if self.instant_train:
                        self.clfs[name].train()

                with ThreadPoolExecutor() as executor:
                    # Submit each classifier training task to the executor
                    futures = [executor.submit(train_single_label, name) for name in clfs]

                    # Wait for all tasks to complete
                    for future in futures:
                        future.result()  # Propagate exceptions if any

                self._needs_training = False
            else:
                LOG.error("Not enough intents registered, at least 2 needed!")

    def eval_fp(self):
        # evaluate false positives, via all unified negative samples
        if len(self.clfs) > 2:
            names = sorted(list(self.clfs))
            for name in names:
                clf = self.clfs[name]
                if not clf.positives:
                    continue
                negatives = []
                for name2 in [n for n in self.clfs if n != name]:
                    negatives += self.clfs[name2].positives
                data: TrainingData = []
                # we don't have a training set for positives
                # but we have a lot of unseen negatives
                negatives = [n for n in negatives if n not in clf.negatives]
                if not negatives:
                    continue
                for s in clf.positives:
                    data.append((clf.one_hot_encode(s), "intent"))
                for s in negatives:
                    data.append((clf.one_hot_encode(s), "not-intent"))
                random.shuffle(data)
                X, Y = zip(*data)
                score = clf.score(X, Y)
                LOG.info(f"TRAINING SCORE: {name}: {score}\n"
                         f"\tN positive samples: {len(clf.positives)}\n"
                         f"\tN negative samples: {len(negatives)}")

    def predict(self, text: str) -> Dict[str, float]:
        """
        Predicts the probabilities for each label.

        Args:
            text (str): The input text.

        Returns:
            Dict[str, float]: A dictionary with label names and their probabilities.
        """
        if self._needs_training:
            self.train()
        # self.eval_fp()  # TODO only for debug
        return {k: clf.predict(text) for k, clf in self.clfs.items()}


if __name__ == "__main__":
    p = TemplateMatcher()
    templs = [
        "change the color to {color}",
        "change light to {color}",
        "change light from {color} to {color2}",
        "{color} is my favorite color",
        "{color} and {color2} are my 2 favorite colors"
    ]
    p.add_templates(templs)

    test = ["red blue and green are my 3 favorite colors",
            "red is my favorite color",
            "light to blue",
            "change to green",
            "make color green"]
    for t in test:
        print(p.match(t))
        # {'slots': {'color2': 'green', 'color': 'red blue'}, 'conf': 0.5622727272727273}
        # {'slots': {'color': 'red'}, 'conf': 0.96}
        # {'slots': {}, 'conf': 0.0}
        # {'slots': {}, 'conf': 0.0}
        # {'slots': {}, 'conf': 0.0}

    intent_1 = ["hello world", "hey there", "hello"]
    intent_2 = ["tell me a joke", "say a joke", "make me laugh"]
    intent_3 = ["how is the weather", "what's the weather like",
                "what is the weather outside"]

    d = DynamicClassifier(instant_train=False)
    d.add_label("hello", intent_1)
    d.add_label("joke", intent_2)
    d.add_label("weather", intent_3)

    for s in ["hello earth", "tell me a joke", "what is the weather", "tell me a joke about the weather"]:
        print(d.predict(s))
        # {'hello': 0.8846260218427711, 'joke': 0.01052660372961578, 'weather': 0.0054173262221566265}
        # {'hello': 0.04226167567958583, 'joke': 0.9994939853998249, 'weather': 0.19370873796804788}
        # {'hello': 0.012769577335433602, 'joke': 0.014340719315762085, 'weather': 0.994151173965159}
        # {'hello': 0.03188504271547432, 'joke': 0.9286979183570826, 'weather': 0.44380980367057876}

    exit()
    c1 = DynamicBinaryClassifier()
    c1.add_positive(intent_1)
    c1.add_negative(intent_2 + intent_3)

    c2 = DynamicBinaryClassifier()
    c2.add_positive(intent_2)
    c2.add_negative(intent_1 + intent_3)

    c3 = DynamicBinaryClassifier()
    c3.add_positive(intent_3)
    c3.add_negative(intent_2 + intent_1)

    for s in ["hello earth", "tell me a joke", "what is the weather", "tell me a joke about the weather"]:
        print(s)
        print("hello", c1.predict(s))
        print("joke", c2.predict(s))
        print("weather", c3.predict(s))
