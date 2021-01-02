import os
import unittest

import numpy as np

from dosma.data_io.med_volume import MedicalVolume
from dosma.data_io.dicom_io import DicomReader

from .. import util as ututils


class TestMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray([
        [0., 0., 0.8, -171.41],
        [0., -0.3125, 0., 96.0154],
        [-0.3125, 0., 0., 47.0233],
        [0., 0., 0., 1.]
    ])  # ('SI', 'AP', 'LR')

    def test_clone(self):
        mv = MedicalVolume(np.random.rand(10,20,30), self._AFFINE)
        mv2 = mv.clone()
        assert mv.is_identical(mv2)  # expected identical volumes
    
        dr = DicomReader(num_workers=ututils.num_workers())
        mv = dr.load(ututils.get_dicoms_path(ututils.get_scan_dirpath("qdess")))[0]
        mv2 = mv.clone(headers=False)
        assert mv.is_identical(mv2)  # expected identical volumes
        assert id(mv.headers) == id(mv2.headers)  # headers not cloned, expected same memory address

        mv3 = mv.clone(headers=True)
        assert mv.is_identical(mv3)  # expected identical volumes
        assert id(mv.headers) != id(mv3.headers)  # headers cloned, expected different memory address


if __name__ == "__main__":
    unittest.main()
