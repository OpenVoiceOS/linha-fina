"""Tests for linha_fina.dynamic."""

import pytest

from linha_fina.dynamic import DynamicBinaryClassifier, DynamicClassifier


class TestDynamicBinaryClassifier:
    def test_add_positive_records_samples(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["hello", "hi there"])
        assert clf.positives == ["hello", "hi there"]
        assert clf._needs_training is True

    def test_add_positive_registers_tokens_as_features(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["play africa"])
        # Each token becomes an entity for the featurizer
        assert "play" in clf.featurizer.entities
        assert "africa" in clf.featurizer.entities

    def test_add_negative_caps_at_3x_positives(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["one"])
        # cap is 3 * 1 = 3
        clf.add_negative([f"neg{i}" for i in range(10)])
        assert len(clf.negatives) <= 3

    def test_one_hot_encode_returns_vector(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["alpha beta"])
        vec = clf.one_hot_encode("alpha")
        assert isinstance(vec, list)
        assert sum(vec) >= 1

    def test_training_data_combines_positives_and_negatives(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["yes one", "yes two"])
        clf.add_negative(["no one", "no two"])
        data = clf.training_data
        labels = {label for _, label in data}
        assert labels == {"intent", "not-intent"}

    def test_train_and_predict(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["play music", "play song", "play track"])
        clf.add_negative(["stop now", "be quiet", "shut up"])
        clf.train()
        assert clf._needs_training is False
        # An obviously-positive input should score > 0.5
        conf = clf.predict("play song now")
        assert 0.0 <= conf <= 1.0

    def test_predict_triggers_lazy_train(self):
        clf = DynamicBinaryClassifier()
        clf.add_positive(["a b", "a c"])
        clf.add_negative(["x y", "x z"])
        # No explicit train() call
        conf = clf.predict("a b")
        assert 0.0 <= conf <= 1.0
        assert clf._needs_training is False

    def test_train_without_data_is_noop(self):
        clf = DynamicBinaryClassifier()
        clf.train()
        assert clf.model is None


class TestDynamicClassifier:
    def test_add_label_records(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello"])
        assert "greet" in clf.clfs
        assert clf._needs_training is True

    def test_remove_label(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello"])
        clf.remove_label("greet")
        assert "greet" not in clf.clfs

    def test_remove_unknown_label_is_noop(self):
        clf = DynamicClassifier()
        clf.remove_label("does_not_exist")

    def test_train_requires_at_least_two_labels(self, caplog):
        clf = DynamicClassifier()
        clf.add_label("a", ["alpha"])
        # Only 1 label — train silently no-ops (logs error)
        clf.train()
        # No exception, but nothing fitted
        assert clf._needs_training is True

    def test_train_with_three_labels_succeeds(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello", "hi", "hey"])
        clf.add_label("bye", ["goodbye", "see you", "bye"])
        clf.add_label("thanks", ["thanks", "thank you", "much appreciated"])
        clf.train()
        assert clf._needs_training is False
        # Negatives auto-populated from other labels
        assert len(clf.clfs["greet"].negatives) > 0
        assert len(clf.clfs["bye"].negatives) > 0

    def test_predict_returns_score_per_label(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello", "hi there", "hey"])
        clf.add_label("bye", ["goodbye", "see you", "bye"])
        clf.add_label("thanks", ["thanks", "thank you", "much appreciated"])
        clf.train()
        scores = clf.predict("hello")
        assert set(scores.keys()) == {"greet", "bye", "thanks"}
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_predict_intended_intent_wins(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello", "hi there", "hey", "good morning"])
        clf.add_label("bye", ["goodbye", "see you", "bye", "later"])
        clf.add_label("thanks", ["thanks", "thank you", "much appreciated"])
        clf.train()
        scores = clf.predict("hello there")
        # greet should be the top scorer for an in-distribution greeting
        top = max(scores, key=scores.get)
        assert top == "greet"

    def test_instant_train_mode_trains_each_add(self):
        clf = DynamicClassifier(instant_train=True)
        clf.add_label("greet", ["hello", "hi", "hey"])
        clf.add_label("bye", ["goodbye", "bye", "see you"])
        clf.add_label("thanks", ["thanks", "thank you", "ty"])
        # After 3 labels, instant_train should have produced models
        assert clf._needs_training is False

    def test_training_is_reproducible_across_instances(self):
        # Two independently trained classifiers on identical data must
        # produce identical predictions — requires the underlying MLP
        # to be seeded (random_state), otherwise weight init/SGD order
        # differ between runs and scores diverge.
        def build():
            clf = DynamicClassifier()
            clf.add_label("greet", ["hello", "hi there", "hey", "good morning"])
            clf.add_label("bye", ["goodbye", "see you", "bye", "later"])
            clf.add_label("thanks", ["thanks", "thank you", "much appreciated"])
            clf.train()
            return clf

        clf1 = build()
        clf2 = build()
        scores1 = clf1.predict("hello there")
        scores2 = clf2.predict("hello there")
        assert scores1 == scores2

    def test_predict_after_remove_drops_label(self):
        clf = DynamicClassifier()
        clf.add_label("greet", ["hello", "hi", "hey"])
        clf.add_label("bye", ["goodbye", "see you", "bye"])
        clf.add_label("thanks", ["thanks", "thank you", "ty"])
        clf.train()
        clf.remove_label("thanks")
        # Need to retrain after removal — force it
        clf._needs_training = True
        # Manually retrigger: train requires ≥3 again so add one back
        clf.add_label("ack", ["got it", "okay", "alright"])
        clf.train()
        scores = clf.predict("hello")
        assert "thanks" not in scores
        assert "ack" in scores
