import numpy as np
from sklearn.datasets import load_iris

class Perceptron(object):

    def __init__(self, shape, learning_rate=0.1, epochs=50):
        self.weight = np.random.randn(1, shape + 1)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def activation(self, X):
        prediction = []
        for elem in X:
            if elem>=0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        prediction = np.asarray(prediction)
        return prediction

    def predict(self, X):
        z = np.dot(X, self.weight[:, 1:].T) + self.weight[0, 0]
        a = self.activation(z)
        return a

    def fit(self, X, d):
        self.errors = []
        for epoch in range(self.epochs):
           y = self.predict(X)
           e = d - y
           mean_error = np.mean(e)
           self.weight[:, 1:] += (self.learning_rate*np.dot(X.T, e)).T
           self.weight[:, 0] += self.learning_rate*mean_error
           err = mean_error**2
           self.errors.append(err)
           print("epoch=%d, error=%.2f" % (epoch, err))

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    X_train = X[:90]
    X_test = X[90:100]
    y = iris.target
    y_train = y[:90]
    y_test = y[90:100]

    perceptron = Perceptron(shape=X.shape[1])
    perceptron.fit(X_train, y_train)
