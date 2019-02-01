import numpy as np
from sklearn.datasets import load_iris

class Perceptron(object):

    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            err = 0
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                err += e
                self.W = self.W + self.lr*e*x
            print("epoch=%d, lrate=%.2f, error=%.2f" % (epoch, self.lr, err))

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target

    perceptron = Perceptron(input_size=4)
    perceptron.fit(X, y)
