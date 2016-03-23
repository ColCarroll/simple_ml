import unittest


class MLTestCase(unittest.TestCase):
    def assertArrayEqual(self, x, y, precision=5):
        self.assertEqual(len(x), len(y))
        for j, (x_j, y_j) in enumerate(zip(x, y)):
            self.assertAlmostEqual(
                x_j,
                y_j,
                precision,
                '{} != {} in index {} to {} decimal places'.format(x, y, j, precision))
