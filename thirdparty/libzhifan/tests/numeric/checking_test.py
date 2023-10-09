import unittest
import numpy as np
import torch

from libzhifan.numeric import *


class CheckingTest(unittest.TestCase):

    def test_checking(self):
        a = torch.ones([1, 2, 3])
        a_ok = (-1, 2, 3)
        a_bad = (-1, 1, 2, 3)
        check_shape(a, a_ok)
        try:
            check_shape(a, a_bad)
        except ValueError as e:
            print("Get correct message: ", e)

        check_shape_equal(a, a)
        b = torch.ones([1])
        try:
            check_shape(a, b)
        except ValueError as e:
            print("Get correct message: ", e)


if __name__ == '__main__':
    unittest.main()
