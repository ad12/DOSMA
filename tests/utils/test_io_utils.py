import os
import shutil
import unittest

import numpy as np
import pandas as pd

from dosma.utils import io_utils

from .. import util

IO_UTILS_DATA = os.path.join(util.UNITTEST_DATA_PATH, "io_utils")


class UtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        io_utils.mkdirs(IO_UTILS_DATA)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(IO_UTILS_DATA)

    def test_h5(self):
        filepath = os.path.join(IO_UTILS_DATA, "sample.h5")
        datas = [{"type": np.random.rand(10, 45, 2), "type2": np.random.rand(13, 95, 4)}]

        for data in datas:
            io_utils.save_h5(filepath, data)
            data2 = io_utils.load_h5(filepath)

            assert len(list(data.keys())) == len(list(data2.keys()))

            for key in data.keys():
                assert (data[key] == data2[key]).all()

    def test_pik(self):
        filepath = os.path.join(IO_UTILS_DATA, "sample.pik")
        datas = {"type": np.random.rand(10, 45, 2), "type2": np.random.rand(13, 95, 4)}

        io_utils.save_pik(filepath, datas)
        datas2 = io_utils.load_pik(filepath)

        for data in datas:
            assert (datas[data] == datas2[data]).all()

    def test_save_tables(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        path = os.path.join(IO_UTILS_DATA, "table.xlsx")
        io_utils.save_tables(path, [df])

        df2 = pd.read_excel(path, engine="openpyxl")
        assert np.all(df == df2)
