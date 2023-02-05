import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000) -> None:
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_prediction = np.dot(x, self.weights) + self.bias
            predictions = sigmoid(linear_prediction)

            dw = (1/num_samples) * np.dot(x.T, (predictions-y))
            db = (1/num_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, x):
        linear_prediction = np.dot(x, self.weights) + self.bias
        y_predictions = sigmoid(linear_prediction)
        class_prediction = [0 if 0.5 else 1 for y in y_predictions]
        return class_prediction