from random import normalvariate


class Array(list):
    def _check(self, other):
        assert len(self) == len(other)

    def __add__(self, other):
        self._check(other)
        return Array([sum(v) for v in zip(self, other)])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, c):
        return Array([c * j for j in self])

    def __rmul__(self, c):
        return self * c

    def __div__(self, c):
        return self * (1. / c)

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return Array(map(abs, self))

    def dot(self, other):
        self._check(other)
        return sum(j * k for j, k in zip(self, other))

    def norm_squared(self):
        return self.dot(self)

    def norm(self):
        return self.norm_squared() ** 0.5


def ones(n):
    """Helper array of ones"""
    return Array([1 for _ in range(n)])


def random(n):
    return Array([normalvariate(0, 1) for _ in range(n)])


def zeros(n):
    return 0 * ones(n)
