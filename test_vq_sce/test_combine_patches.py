import matplotlib.pyplot as plt
import numpy as np
import os
import unittest

from syntheticcontrast_v02.utils.combine_patches import CombinePatches


class TestCombinePatches(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.EMPTY_FOLDER = "./FixturesUnpairedEmpty/"
        self.TEST_FOLDER = "./FixturesUnpaired/"

        self.test_config = {"data": {"stride_length": 16}}

    def tearDown(self) -> None:
        super().tearDown()


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Quick routine to visually check output of CombinePatches """

    test_config = {"data": {"stride_length": 16}}

    d = CombinePatches(test_config)
