import unittest
import numpy as np
import torch

from libzhifan import quick_inspect


class QuickInspectTest(unittest.TestCase):

    def test(self):
        pyobj = [
            dict(a=[1, np.zeros([4, 3])]),
            torch.ones([3, 4])]
        quick_inspect(pyobj)


if __name__ == '__main__':
    unittest.main()
