from linear_algebra import Array, zeros, ones


class Function(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __mul__(self, c):
        return Function(lambda x: c * self(x))

    def __rmul__(self, c):
        return self * c

    def __add__(self, other):
        return Function(lambda x: self(x) + other(x))

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __div__(self, c):
        return self * (1. / c)

    def __rdiv__(self, c):
        return self * (1. / c)

    def grad(self, x):
        return Array([derivative(self, x, eps(x, j)) for j in range(len(x))])

    def minimize(self, dims=3, tol=0.0000000001):
        step = 0.1
        x_new, x_old = ones(dims), zeros(dims)
        err_new = (x_new - x_old).norm()
        while err_new > tol:
            x_new, x_old = x_new - (step * self.grad(x_new)), x_new
            err_new, err_old = (x_new - x_old).norm(), err_new
            if err_old < err_new:
                step *= 0.5
                x_new, err_new = x_old, err_old
        return x_new


def eps(x, idx, size=0.0000001):
    ary = zeros(len(x))
    ary[idx] = size
    return ary


def derivative(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2 * eps.norm())
