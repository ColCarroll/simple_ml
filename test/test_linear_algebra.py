from linear_algebra import Array
from test import MLTestCase


class TestArray(MLTestCase):
    def setUp(self):
        self.x = Array([1, 2, 3])

    def test__check(self):
        with self.assertRaises(AssertionError):
            self.x + Array([1, 2])

    def test_addition(self):
        self.assertEqual(self.x + Array([1, 1, 1]), Array([2, 3, 4]))

    def test_subtraction(self):
        self.assertEqual(self.x - self.x, Array([0, 0, 0]))

    def test_multiplication(self):
        self.assertEqual(2 * self.x, Array([2, 4, 6]))
        self.assertEqual(self.x * -1, Array([-1, -2, -3]))

    def test_neg(self):
        self.assertEqual(-self.x + self.x, Array([0, 0, 0]))

    def test_dot(self):
        self.assertEqual(self.x.dot(Array([2, 4, 6])), 1 * 2 + 2 * 4 + 3 * 6)

    def test_norm(self):
        self.assertEqual(self.x.norm(), (1 ** 2 + 2 ** 2 + 3 ** 2) ** 0.5)

    def test_get_set_item(self):
        x = Array([1, 1, 1])
        self.assertEqual(x[2], 1)
        x[2] = 8
        self.assertEqual(x[2], 8)
