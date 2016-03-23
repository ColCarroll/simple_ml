from calculus import Function
from linear_algebra import Array
from test import MLTestCase


class TestFunction(MLTestCase):
    def setUp(self):
        self.x = Array([1, 2, 3])
        self.f = Function(lambda x: x.norm())

    def test_call(self):
        self.assertEqual(self.f(self.x), self.x.norm())

    def test_mul(self):
        g = 3 * self.f
        h = self.f * -1
        self.assertEqual(g(self.x), 3 * self.x.norm())
        self.assertEqual(h(self.x), -self.x.norm())

    def test_add(self):
        double_norm = self.f + self.f
        self.assertEqual(double_norm(self.x), 2 * self.x.norm())

    def test_sub(self):
        self.assertEqual((self.f - self.f)(self.x), 0)

    def test_grad(self):
        # D(||x||) = x / ||x||
        df = Function(lambda x: x.norm()).grad(self.x)
        expected = self.x / self.x.norm()
        self.assertArrayEqual(df, expected)

        # D(||x||^2) = 2 * x
        df = Function(lambda x: x.norm() ** 2).grad(self.x)
        expected = 2 * self.x
        self.assertArrayEqual(df, expected)

        # D(x[0]^2 - x[1] * x[2]) = (2 * x[0], -x[2], -x[1])
        df = Function(lambda x: x[0] ** 2 - x[1] * x[2]).grad(self.x)
        expected = Array([2 * self.x[0], -self.x[2], -self.x[1]])
        self.assertArrayEqual(df, expected)

    def test_minimize(self):
        # ||x - v||^2 is minimized when x = v
        f = Function(lambda x: (x - Array(range(len(x)))).norm_squared())
        for dims in range(1, 3):
            self.assertArrayEqual(f.minimize(dims=dims), Array(range(dims)), 3)
