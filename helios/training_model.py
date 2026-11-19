"""Beginner-friendly Logistic Regression for HELIOS fraud detection.

Use scikit-learn if it imports successfully. Otherwise, train the same kind of
model using plain Python. Both options save weights as ordinary JSON numbers.
"""

import math
import os
import warnings


def sigmoid(number):
    """Turn a model score into a number between 0 and 1."""
    number = max(-35, min(35, number))  # avoid math.exp overflow
    return 1 / (1 + math.exp(-number))


class FraudModel:
    def __init__(self, epochs=260, rate=0.07):
        self.epochs = epochs
        self.rate = rate
        self.threshold = 0.80  # Predict fraud if probability >= 80%
        self.backend = "pure_python"
        self.weights = []
        self.means = []
        self.scales = []

    def _scaled(self, transaction):
        """Standardize each feature so money and percentages are comparable."""
        return [
            (float(value) - mean) / scale
            for value, mean, scale in zip(transaction, self.means, self.scales)
        ]

    def fit(self, transactions, labels):
        """Learn from labeled examples, with scikit-learn or plain Python."""
        if not transactions or len(transactions) != len(labels):
            raise ValueError("Need labeled training data")
        if set(labels) != {0, 1}:
            raise ValueError("Both fraud and legitimate examples are required")

        # STEP 1: Learn how to normalize the six numerical input features.
        columns = list(zip(*transactions))
        self.means = [sum(column) / len(column) for column in columns]
        self.scales = []
        for column, mean in zip(columns, self.means):
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            self.scales.append(max(math.sqrt(variance), 0.001))
        scaled_data = [self._scaled(row) for row in transactions]

        # STEP 2: Try the scikit-learn version used in the original project.
        choice = os.getenv("HELIOS_ML_BACKEND", "auto").strip().lower()
        if choice not in {"auto", "sklearn", "python"}:
            raise ValueError("HELIOS_ML_BACKEND must be auto, sklearn or python")
        if choice != "python":
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",  # fraud is less common than legitimate
                    random_state=42,
                )
                classifier.fit(scaled_data, labels)
                self.weights = [float(classifier.intercept_[0])]
                self.weights += [float(value) for value in classifier.coef_[0]]
                self.backend = "sklearn"
                return self
            except (ImportError, OSError) as error:
                # A managed Windows PC might block an sklearn DLL.
                # Never ask users to disable security to run this project.
                print(f"scikit-learn unavailable ({error}). Using pure Python instead.")

        # STEP 3: Fallback: gradient descent written only with Python + math.
        feature_count = len(self.means)
        self.weights = [0.0] * (feature_count + 1)  # bias + feature weights
        self.backend = "pure_python"
        fraud_count = sum(labels)
        legitimate_count = len(labels) - fraud_count
        fraud_weight = min(10.0, legitimate_count / fraud_count)

        for _ in range(self.epochs):
            gradients = [0.0] * len(self.weights)
            for row, correct_label in zip(scaled_data, labels):
                score = self.weights[0]
                score += sum(w * x for w, x in zip(self.weights[1:], row))
                prediction = sigmoid(score)
                error = (prediction - correct_label)
                if correct_label == 1:
                    error *= fraud_weight  # learn more from rare fraudulent examples
                gradients[0] += error
                for feature_index, value in enumerate(row):
                    gradients[feature_index + 1] += error * value

            # Update the weights to reduce prediction error.
            for index, gradient in enumerate(gradients):
                penalty = 0 if index == 0 else 0.002 * self.weights[index]
                self.weights[index] -= self.rate * (
                    gradient / len(scaled_data) + penalty
                )
        return self

    def predict_proba(self, transactions):
        """Return [probability_of_legitimate, probability_of_fraud]."""
        probabilities = []
        for transaction in transactions:
            row = self._scaled(transaction)
            score = self.weights[0]
            score += sum(w * x for w, x in zip(self.weights[1:], row))
            fraud_probability = sigmoid(score)
            probabilities.append([1 - fraud_probability, fraud_probability])
        return probabilities

    def predict(self, transactions):
        """Return 1 for fraud or 0 for legitimate."""
        return [
            int(probabilities[1] >= self.threshold)
            for probabilities in self.predict_proba(transactions)
        ]

    def to_dict(self):
        """Store the trained model in JSON, without joblib or sklearn DLLs."""
        return {
            "algorithm": "portable_logistic_regression",
            "backend": self.backend,
            "epochs": self.epochs,
            "rate": self.rate,
            "weights": self.weights,
            "means": self.means,
            "scales": self.scales,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, saved):
        """Load saved model, including older pure-Python HELIOS versions."""
        if saved.get("algorithm") not in {
            "portable_logistic_regression",
            "pure_python_weighted_logistic_regression",
        }:
            raise ValueError("Unsupported model format")
        model = cls(int(saved["epochs"]), float(saved["rate"]))
        model.weights = [float(value) for value in saved["weights"]]
        model.means = [float(value) for value in saved["means"]]
        model.scales = [float(value) for value in saved["scales"]]
        model.threshold = float(saved["threshold"])
        model.backend = saved.get("backend", "pure_python")
        if len(model.weights) != len(model.means) + 1:
            raise ValueError("Invalid model weights")
        if len(model.scales) != len(model.means):
            raise ValueError("Invalid model feature scaling")
        return model
