import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000) -> None:
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent 
        for _ in range(self.num_iterations):
            y_prediction = np.dot(x, self.weights) + self.bias

            dw = (1 / num_samples) * np.dot(x.T, (y_prediction - y))
            db = (1 / num_samples) * np.sum(y_prediction - y)

            # updating the bias and weights simultaneously after each iteration
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, x):
        y_prediction = np.dot(x, self.weights) + self.bias
        return y_prediction


