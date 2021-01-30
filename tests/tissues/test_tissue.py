import unittest

import numpy as np

from dosma.tissues.tissue import largest_cc


class TestLargestCC(unittest.TestCase):
    def test_largest_cc(self):
        # smallest cc
        a = np.zeros((100, 100)).astype(np.uint8)
        a[:10, :10] = 1

        # medium cc
        b = np.zeros((100, 100)).astype(np.uint8)
        b[85:, 85:] = 1

        # largest cc
        c = np.zeros((100, 100)).astype(np.uint8)
        c[25:75, 25:75] = 1

        mask = a | b | c

        assert np.all(largest_cc(mask) == c)  # only largest cc returned
        assert np.all(largest_cc(mask, num=2) == (b | c))  # largest 2 cc
        assert np.all(largest_cc(mask, num=3) == (a | b | c))  # largest 3 cc
        assert np.all(largest_cc(mask, num=4) == (a | b | c))  # only 3 cc, return all


if __name__ == "__main__":
    unittest.main()
