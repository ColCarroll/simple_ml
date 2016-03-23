from math import exp, log

from calculus import Function
from linear_algebra import Array


def sum_squared_error_function(X, y):
    def squared_error(w):
        Xw = Array([x.dot(w) for x in X])
        return (Xw - y).norm()
    return squared_error


def log_loss_error_function(X, y):
    def log_loss(w):
        return -sum((1 - y_j) * log(1 - x) + y_j * log(x) for x, y_j in zip(X, y))
    return log_loss


class LinearRegression(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        dims = len(X[0])
        self.coef_ = Function(sum_squared_error_function(X, y)).minimize(dims=dims)

    def transform(self, X):
        if self.coef_ is None:
            raise NotImplementedError("Must fit the model first")
        return Array([self.coef_.dot(x) for x in X])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class LogisticRegression(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        dims = len(X[0])
        self.coef_ = Function(log_loss_error_function(X, y)).minimize(dims=dims)

    def transform(self, X):
        if self.coef_ is None:
            raise NotImplementedError("Must fit the model first")
        return Array([1 / (1 + exp(-self.coef_.dot(x))) for x in X])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
