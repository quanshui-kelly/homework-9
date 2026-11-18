############################################################
# CIS 521: Homework 9
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import homework9_data as data

############################################################

student_name = "Xinyuan Quan"

############################################################
# Section 1: Perceptrons
############################################################


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        examples = list(examples)
        self.weights = {}

        for _ in range(iterations):
            for features, label in examples:
                prediction = self.predict(features)

                if prediction != label:
                    direction = 1.0 if label else -1.0

                    for name, value in features.items():
                        old_weight = self.weights.get(name, 0.0)
                        self.weights[name] = (
                            old_weight + direction * value
                        )

    def predict(self, x):
        def _dot_product(weights, features):
            return sum(weights.get(name, 0.0) * value
                    for name, value in features.items())
        return _dot_product(self.weights, x) > 0


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        examples = list(examples)
        self.labels = list(dict.fromkeys(label for _, label in examples))
        self.weights = {label: {} for label in self.labels}

        for _ in range(iterations):
            for features, true_label in examples:
                predicted_label = self.predict(features)

                if predicted_label != true_label:
                    true_weights = self.weights[true_label]
                    predicted_weights = self.weights[predicted_label]

                    for name, value in features.items():
                        true_weights[name] = (
                            true_weights.get(name, 0.0) + value
                        )
                        predicted_weights[name] = (
                            predicted_weights.get(name, 0.0) - value
                        )
    
    def predict(self, x):
        def _dot_product(weights, features):
            return sum(weights.get(name, 0.0) * value
                    for name, value in features.items())
        
        if not self.labels:
            raise ValueError("Cannot predict without any class labels.")

        return max(
            self.labels,
            key=lambda label: _dot_product(self.weights[label], x),
        )

############################################################
# Section 2: Applications
############################################################


class IrisClassifier(object):

    def _features(instance):
        features = {index: value
                    for index, value in enumerate(instance)}
        features["bias"] = 1.0
        return features

    def __init__(self, data):
        examples = [
            (self._features(instance), label)
            for instance, label in data
        ]
        self.classifier = MulticlassPerceptron(examples, 50)

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class DigitClassifier(object):

    def _features(instance):
        features = {index: value
                    for index, value in enumerate(instance)}
        features["bias"] = 1.0
        return features

    def __init__(self, data):
        examples = [
            (self._features(instance), label)
            for instance, label in data
        ]
        self.classifier = MulticlassPerceptron(examples, 10)

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class BiasClassifier(object):

    def _features(instance):
        return {"value": instance, "bias": 1.0}

    def __init__(self, data):
        examples = [
            (self._features(instance), label)
            for instance, label in data
        ]
        self.classifier = BinaryPerceptron(examples, 10)

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class MysteryClassifier1(object):

    def _features(instance):
        x_1, x_2 = instance
        radius_squared = x_1 ** 2 + x_2 ** 2
        return {"radius_squared": radius_squared, "bias": 1.0}

    def __init__(self, data):
        examples = [
            (self._features(instance), label)
            for instance, label in data
        ]
        self.classifier = BinaryPerceptron(examples, 50)

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class MysteryClassifier2(object):

    def _features(instance):
        x_1, x_2, x_3 = instance
        return {"product": x_1 * x_2 * x_3}

    def __init__(self, data):
        examples = [
            (self._features(instance), label)
            for instance, label in data
        ]
        self.classifier = BinaryPerceptron(examples, 1)

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))

############################################################
# Section 3: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
6
"""

feedback_question_2 = """
The most challenging part of this assignment was choosing appropriate 
feature transformations for the two mystery data sets. 
"""

feedback_question_3 = """
I enjoyed the mystery-classifier questions because they showed that 
choosing a good representation can be just as important as 
choosing a learning algorithm.
"""
