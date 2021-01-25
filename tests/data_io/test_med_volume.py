import os
import unittest

import numpy as np
import SimpleITK as sitk

from dosma.data_io.med_volume import MedicalVolume
from dosma.data_io.dicom_io import DicomReader
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.nifti_io import NiftiReader

from .. import util as ututils


class TestMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray([
        [0., 0., 0.8, -171.41],
        [0., -0.3125, 0., 96.0154],
        [-0.3125, 0., 0., 47.0233],
        [0., 0., 0., 1.]
    ])  # ('SI', 'AP', 'LR')

    def test_reformat(self):
        mv = MedicalVolume(np.random.rand(10,20,30), self._AFFINE)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])
        mv2 = mv.reformat(new_orientation)
        assert mv2.orientation == new_orientation
        assert id(mv) == id(mv2)

    def test_reformat_as(self):
        mv = MedicalVolume(np.random.rand(10,20,30), self._AFFINE)
        mv2 = MedicalVolume(np.random.rand(10,20,30), self._AFFINE[:, (0,2,1,3)])
        mv = mv.reformat_as(mv2)
        assert mv.orientation == mv2.orientation

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
    
    def test_to_sitk(self):
        filepath = ututils.get_read_paths(ututils.get_scan_dirpath("qdess"), ImageDataFormat.nifti)[0]
        expected = sitk.ReadImage(filepath)
        
        nr = NiftiReader()
        mv = nr.load(filepath)
        img = mv.to_sitk()

        assert np.allclose(sitk.GetArrayViewFromImage(img), sitk.GetArrayViewFromImage(expected))
        assert np.allclose(img.GetOrigin(), expected.GetOrigin())
        assert img.GetSpacing() == img.GetSpacing()
        assert img.GetDirection() == expected.GetDirection()
    
    def test_from_sitk(self):
        filepath = ututils.get_read_paths(ututils.get_scan_dirpath("qdess"), ImageDataFormat.nifti)[0]
        nr = NiftiReader()
        expected = nr.load(filepath)

        img = sitk.ReadImage(filepath)
        mv = MedicalVolume.from_sitk(img)

        assert np.allclose(mv.affine, expected.affine)
        assert mv.volume.shape == expected.volume.shape
        assert np.all(mv.volume == expected.volume)

    def test_math(self):
        mv1 = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10,20,30)), self._AFFINE)

        out = mv1 + mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 3)

        out = mv1 - mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == -1)

        out = mv1 * mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 2)

        out = mv1 / mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 0.5)

        out = mv1 ** mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 1)

        out = mv1.clone()
        out += mv2
        assert np.all(out._volume == 3)

        out = mv1.clone()
        out -= mv2
        assert np.all(out._volume == -1)

        out = mv1.clone()
        out *= mv2
        assert np.all(out._volume == 2)

        out = mv1.clone()
        out /= mv2
        assert np.all(out._volume == 0.5)

        out = mv1.clone()
        out **= mv2
        assert np.all(out._volume == 1)

        mv3 = mv1.clone().reformat(mv1.orientation[::-1])
        with self.assertRaises(ValueError):
            mv3 + mv2

    def test_comparison(self):
        mv1 = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10,20,30)), self._AFFINE)

        assert np.all((mv1 == mv1.clone()).volume)
        assert np.all((mv1 != mv2).volume)
        assert np.all((mv1 < mv2).volume)
        assert np.all((mv1 <= mv2).volume)
        assert np.all((mv2 > mv1).volume)
        assert np.all((mv2 >= mv1).volume)


if __name__ == "__main__":
    unittest.main()
