from linear_algebra import Array
from machine_learning import LinearRegression
from test import MLTestCase


class TestLinearRegression(MLTestCase):
    def test_no_noise(self):
        test_X = [Array([1, j, j ** 2]) for j in range(10)]
        weights = Array([1, 0.2, 1])
        test_y = Array([weights.dot(x) for x in test_X])
        reg = LinearRegression()
        reg.fit(test_X, test_y)
        self.assertArrayEqual(reg.coef_, weights, 2)
