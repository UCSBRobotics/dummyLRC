import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.5, random_state = 0)
y_test = y_test.astype(bool).astype("i1")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LRC:
    def __init__(self, step_size):
        self.weights = np.zeros(64)
        self.step_size = step_size
    def predict(self, input):
        z = np.dot(self.weights, input)
        return sigmoid(z)
    def train(self, data, labels, epochs):
        for i in range(epochs):
            for example, label in zip(data, labels):
                #w_new = w_old + learn_rate(label-prediction)*input
                self.weights += self.step_size * (label - self.predict(example)) * example

    def eval(self, test, labels):
        correct = 0
        for example, label in zip(test, labels):
            p = self.predict(example)
            p = 0 if p< 0.5 else 1
            if p == label: 
                correct += 1
        print(correct, len(labels), correct/len(labels))
