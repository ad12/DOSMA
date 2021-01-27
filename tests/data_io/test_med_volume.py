import os
import shutil
import unittest

import numpy as np
import SimpleITK as sitk

from dosma.data_io.med_volume import MedicalVolume
from dosma.data_io.dicom_io import DicomReader
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.nifti_io import NiftiReader
from dosma.utils.device import Device

from .. import util as ututils


class TestMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray([
        [0., 0., 0.8, -171.41],
        [0., -0.3125, 0., 96.0154],
        [-0.3125, 0., 0., 47.0233],
        [0., 0., 0., 1.]
    ])  # ('SI', 'AP', 'LR')

    _TEMP_PATH = os.path.join(ututils.TEMP_PATH, __name__)

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls._TEMP_PATH, exist_ok=True)

    @classmethod
    def tearDownCls(cls):
        if os.path.isdir(cls._TEMP_PATH):
            shutil.rmtree(cls._TEMP_PATH)

    def test_reformat(self):
        mv = MedicalVolume(np.random.rand(10,20,30), self._AFFINE)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2.orientation == new_orientation
        assert id(mv2) != id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(new_orientation, inplace=True)
        assert mv2.orientation == new_orientation
        assert id(mv2) == id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(mv.orientation)
        assert id(mv2) != id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(mv.orientation, inplace=True)
        assert id(mv2) == id(mv)
        assert np.shares_memory(mv2._volume, mv._volume)

        mv2 = mv.reformat(new_orientation).reformat(mv.orientation)
        assert mv2.is_identical(mv)

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
        assert img.GetSize() == mv.shape
        assert np.allclose(img.GetOrigin(), expected.GetOrigin())
        assert img.GetSpacing() == img.GetSpacing()
        assert img.GetDirection() == expected.GetDirection()

        mv = MedicalVolume(np.zeros((10,20,1,3)), affine=self._AFFINE)
        img = mv.to_sitk(vdim=-1)
        assert np.all(sitk.GetArrayViewFromImage(img) == 0)
        assert img.GetSize() == (10,20,1)

    def test_from_sitk(self):
        filepath = ututils.get_read_paths(ututils.get_scan_dirpath("qdess"), ImageDataFormat.nifti)[0]
        nr = NiftiReader()
        expected = nr.load(filepath)

        img = sitk.ReadImage(filepath)
        mv = MedicalVolume.from_sitk(img)

        assert np.allclose(mv.affine, expected.affine)
        assert mv.shape == expected.shape
        assert np.all(mv.volume == expected.volume)

        img = sitk.Image([10, 20, 1], sitk.sitkVectorFloat32, 3)
        mv = MedicalVolume.from_sitk(img)
        assert np.all(mv.volume == 0)
        assert mv.shape == (10, 20, 1, 3)

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

    def test_slice(self):
        mv = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        with self.assertRaises(IndexError):
            mv[4]
        mv_slice = mv[4:5]
        assert mv_slice.shape == (1,20,30)

        mv = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        mv[:5,...] = 2
        assert np.all(mv._volume[:5,...] == 2) & np.all(mv._volume[5:,...] == 1)
        assert np.all(mv[:5,...].volume == 2)

        mv = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        mv2 = mv[:5,...].clone()
        mv2 = 2
        mv[:5,...] = mv2
        assert np.all(mv._volume[:5,...] == 2) & np.all(mv._volume[5:,...] == 1)
        assert np.all(mv[:5,...].volume == 2)

    def test_4d(self):
        vol = np.stack([np.ones((10,20,30)), 2*np.ones((10,20,30))], axis=-1)
        mv = MedicalVolume(vol, self._AFFINE)
        assert mv.orientation == ("SI", "AP", "LR")
        assert mv.shape == (10, 20, 30, 2)

        assert np.all(mv[..., 0].volume == 1)
        assert np.all(mv[..., 1].volume == 2)

        ornt = ("AP", "IS", "RL")
        mv2 = mv.reformat(ornt)
        mv2.orientation == ornt
        assert mv2.shape == (20,10,30,2)

        mv2 = mv.reformat(ornt).reformat(mv.orientation)
        assert mv2.is_identical(mv)

        fp = os.path.join(self._TEMP_PATH, "test_4d.nii.gz")
        mv.save_volume(fp)
        mv2 = NiftiReader().load(fp)
        assert mv2.is_identical(mv)

    @ututils.requires_packages("cupy")
    def test_device(self):
        import cupy as cp

        mv = MedicalVolume(np.ones((10,20,30)), self._AFFINE)
        mv_gpu = mv.to(Device(0))

        assert mv_gpu.device == Device(0)
        assert isinstance(mv_gpu.volume, cp.ndarray)
        assert isinstance(mv_gpu.affine, np.ndarray)

        assert mv_gpu.is_same_dimensions(mv)

        assert cp.all((mv_gpu + 1).volume == 2)
        assert cp.all((mv_gpu - 1).volume == 0)
        assert cp.all((mv_gpu * 2).volume == 2)
        assert cp.all((mv_gpu / 2).volume == 0.5)
        assert cp.all((mv_gpu > 0).volume)
        assert cp.all((mv_gpu >= 0).volume)
        assert cp.all((mv_gpu < 2).volume)
        assert cp.all((mv_gpu <= 2).volume)

        ornt = tuple(x[::-1] for x in mv_gpu.orientation[::-1])
        mv2 = mv_gpu.reformat(ornt)
        assert mv2.orientation == ornt

        mv_cpu = mv_gpu.cpu()
        assert mv_cpu.device == Device(-1)
        assert mv_cpu.is_identical(mv)

        with self.assertRaises(RuntimeError):
            mv_gpu.save_volume(os.path.join(self._TEMP_PATH, "test_device.nii.gz"))


if __name__ == "__main__":
    unittest.main()
