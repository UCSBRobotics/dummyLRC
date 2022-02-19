import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNet:
    def __init__(self, step_size, dims):
        self.weights = []
        self.bs = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.weights.append(np.random.rand(b, a)-0.5)
            self.bs.append(np.random.rand(b)-0.5)

        self.step_size = step_size
    def train(self, data, labels, epochs):
        pass
        for i in range(epochs):
            for example, label in zip(data, labels):
                #w_new = w_old + learn_rate(label-prediction)*input
                self.weights += self.step_size * (label - self.predict(example)) * example
    def eval(self, test, labels):
        pass
        correct = 0
        for example, label in zip(test, labels):
            p = self.predict(example)
            p = 0 if p< 0.5 else 1
            if p == label: 
                correct += 1
        print(correct, len(labels), correct/len(labels))

    def forward_prop(self, iput):
        self.zs = []
        self.aus = []
        count = 0
        for w, b in zip(self.weights, self.bs):
            print(w)
            print(iput)

            z = np.matmul(w, iput) + b
            self.zs.append(z)
            if count == len(self.weights) - 1:
                iput = self.softmax(iput)
            else:
                iput = sigmoid(z)
                self.aus.append(iput)
            count+=1
        return iput

    def loss(self, y, y_hat):
        return -np.dot(y, np.log2(y_hat))
    def softmax(self, iput):
        return np.exp(iput) / np.sum(np.exp(iput))