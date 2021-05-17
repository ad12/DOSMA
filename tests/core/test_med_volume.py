import os
import shutil
import unittest

import h5py
import nibabel as nib
import nibabel.testing as nib_testing
import numpy as np
import SimpleITK as sitk

from dosma.core.device import Device
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter
from dosma.core.med_volume import MedicalVolume

from .. import util as ututils


class TestMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray(
        [
            [0.0, 0.0, 0.8, -171.41],
            [0.0, -0.3125, 0.0, 96.0154],
            [-0.3125, 0.0, 0.0, 47.0233],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # ('SI', 'AP', 'LR')

    _TEMP_PATH = os.path.join(ututils.TEMP_PATH, __name__)

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls._TEMP_PATH, exist_ok=True)

    @classmethod
    def tearDownCls(cls):
        if os.path.isdir(cls._TEMP_PATH):
            shutil.rmtree(cls._TEMP_PATH)

    def test_reformat(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
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
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        mv2 = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE[:, (0, 2, 1, 3)])
        mv = mv.reformat_as(mv2)
        assert mv.orientation == mv2.orientation

    def test_reformat_header(self):
        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers(volume.shape[2:])
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2._headers.shape == (30, 1, 1, 40)

        mv2 = mv.clone()
        mv2.reformat(new_orientation, inplace=True)
        assert mv2._headers.shape == (30, 1, 1, 40)

        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers((volume.shape[2], 1))
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)
        new_orientation = tuple(x[::-1] for x in mv.orientation[::-1])

        mv2 = mv.reformat(new_orientation)
        assert mv2._headers.shape == (30, 1, 1, 1)

    def test_metadata(self):
        field, field_val = "EchoTime", 4.0

        volume = np.random.rand(10, 20, 30, 40)
        headers = ututils.build_dummy_headers(volume.shape[2:], {field: field_val})
        mv_no_headers = MedicalVolume(volume, self._AFFINE)
        mv = MedicalVolume(volume, self._AFFINE, headers=headers)

        assert mv_no_headers.headers() is None
        assert mv_no_headers.headers(flatten=True) is None

        with self.assertRaises(ValueError):
            mv.get_metadata("foobar")
        assert mv.get_metadata("foobar", default=0) == 0

        echo_time = mv.get_metadata(field)
        assert echo_time == field_val

        new_val = 5.0
        mv2 = mv.clone(headers=True)
        mv2.set_metadata(field, new_val)
        assert mv.get_metadata(field, type(field_val)) == field_val
        assert mv2.get_metadata(field, type(new_val)) == new_val
        for h in mv2.headers(flatten=True):
            assert h[field].value == new_val

        new_val = 6.0
        mv2 = mv.clone(headers=True)
        mv2[..., 1].set_metadata(field, new_val)
        assert mv2[..., 0].get_metadata(field) == field_val
        assert mv2[..., 1].get_metadata(field) == new_val
        headers = mv2.headers()
        for h in headers[..., 0].flatten():
            assert h[field].value == field_val
        for h in headers[..., 1].flatten():
            assert h[field].value == new_val

        # Set metadata when volume has no headers.
        mv_nh = MedicalVolume(volume, self._AFFINE, headers=None)
        with self.assertRaises(ValueError):
            mv_nh.set_metadata("EchoTime", 40.0)
        with self.assertWarns(UserWarning):
            mv_nh.set_metadata("EchoTime", 40.0, force=True)
        assert mv_nh._headers.shape == (1,) * len(mv_nh.shape)
        assert mv_nh.get_metadata("EchoTime") == 40.0
        assert mv_nh[:1, :2, :3]._headers.shape == (1,) * len(mv_nh.shape)

    def test_clone(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        mv2 = mv.clone()
        assert mv.is_identical(mv2)  # expected identical volumes

        mv = MedicalVolume(
            np.random.rand(10, 20, 30),
            self._AFFINE,
            headers=ututils.build_dummy_headers((1, 1, 30)),
        )
        mv2 = mv.clone(headers=False)
        assert mv.is_identical(mv2)  # expected identical volumes
        assert id(mv.headers(flatten=True)[0]) == id(
            mv2.headers(flatten=True)[0]
        ), "headers not cloned, expected same memory address"

        mv3 = mv.clone(headers=True)
        assert mv.is_identical(mv3)  # expected identical volumes
        assert id(mv.headers(flatten=True)[0]) != id(
            mv3.headers(flatten=True)[0]
        ), "headers cloned, expected different memory address"

    def test_to_nib(self):
        arr = np.random.rand(10, 20, 30)
        mv = MedicalVolume(arr, self._AFFINE)
        nib_img = nib.Nifti1Image(arr, mv.affine)

        nib_from_mv = mv.to_nib()
        assert np.all(nib_from_mv.get_fdata() == nib_img.get_fdata())
        assert np.all(nib_from_mv.affine == nib_img.affine)

    def test_to_sitk(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        filepath = os.path.join(ututils.TEMP_PATH, "med_vol_to_sitk.nii.gz")
        NiftiWriter().save(mv, filepath)

        expected = sitk.ReadImage(filepath)

        nr = NiftiReader()
        mv = nr.load(filepath)
        img = mv.to_sitk()

        assert np.allclose(sitk.GetArrayViewFromImage(img), sitk.GetArrayViewFromImage(expected))
        assert img.GetSize() == mv.shape
        assert np.allclose(img.GetOrigin(), expected.GetOrigin())
        assert img.GetSpacing() == img.GetSpacing()
        assert img.GetDirection() == expected.GetDirection()

        mv = MedicalVolume(np.zeros((10, 20, 1, 3)), affine=self._AFFINE)
        img = mv.to_sitk(vdim=-1)
        assert np.all(sitk.GetArrayViewFromImage(img) == 0)
        assert img.GetSize() == (10, 20, 1)

    def test_from_nib(self):
        filepath = os.path.join(nib_testing.data_path, "example4d.nii.gz")
        nib_img = nib.load(filepath)

        mv = MedicalVolume.from_nib(nib_img)
        assert np.all(mv.affine == nib_img.affine)
        assert np.all(mv.A == nib_img.get_fdata())

        precision = 4
        mv2 = MedicalVolume.from_nib(
            nib_img, affine_precision=precision, origin_precision=precision
        )
        assert np.allclose(mv2.affine, nib_img.affine, atol=10 ** (-precision))
        assert np.all(mv2.A == nib_img.get_fdata())

    def test_from_sitk(self):
        mv = MedicalVolume(np.random.rand(10, 20, 30), self._AFFINE)
        filepath = os.path.join(ututils.TEMP_PATH, "med_vol_from_sitk.nii.gz")
        NiftiWriter().save(mv, filepath)

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
        mv1 = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10, 20, 30)), self._AFFINE)

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

        out = mv1 // mv2
        assert np.all(mv1._volume == 1)
        assert np.all(mv2._volume == 2)
        assert np.all(out._volume == 0)

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
        out //= mv2
        assert np.all(out._volume == 0)

        out = mv1.clone()
        out **= mv2
        assert np.all(out._volume == 1)

        mv3 = mv1.clone().reformat(mv1.orientation[::-1])
        with self.assertRaises(ValueError):
            mv3 + mv2

    def test_numpy(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        assert np.all(np.exp(mv.volume) == np.exp(mv).volume)

        mv[np.where(mv == 1)] = 5
        assert np.all(mv == 5)
        assert not np.any(mv == 1)
        assert all(mv == 5)
        assert not any(mv == 5)

        _ = mv + np.ones(mv.shape)

        shape = (10, 20, 30, 2)
        headers = np.stack(
            [
                ututils.build_dummy_headers(shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )

        # Reduce functions
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4), headers=headers)

        mv2 = np.add.reduce(mv, -1)
        assert np.all(mv2 == np.add.reduce(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.add.reduce(mv, axis=None)
        assert np.all(mv2 == np.add.reduce(mv.volume, axis=None))
        assert np.isscalar(mv2)

        mv2 = np.sum(mv, axis=-1)
        assert np.all(mv2 == np.sum(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.sum(mv)
        assert np.all(mv2 == np.sum(mv.volume))
        assert np.isscalar(mv2)

        mv2 = mv.sum(axis=-1)
        assert np.all(mv2 == np.sum(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = mv.sum()
        assert np.all(mv2 == np.sum(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.mean(mv, axis=-1)
        assert np.all(mv2 == np.mean(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.mean(mv)
        assert np.all(mv2 == np.mean(mv.volume))
        assert np.isscalar(mv2)

        mv2 = mv.mean(axis=-1)
        assert np.all(mv2 == np.mean(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = mv.mean()
        assert np.all(mv2 == np.mean(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.std(mv, axis=-1)
        assert np.all(mv2 == np.std(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.std(mv)
        assert np.all(mv2 == np.std(mv.volume))
        assert np.isscalar(mv2)

        # Min/max functions
        mv2 = np.amin(mv, axis=-1)
        assert np.all(mv2 == np.amin(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.amin(mv)
        assert np.all(mv2 == np.amin(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.amax(mv, axis=-1)
        assert np.all(mv2 == np.amax(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.amax(mv)
        assert np.all(mv2 == np.amax(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.argmin(mv, axis=-1)
        assert np.all(mv2 == np.argmin(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.argmin(mv)
        assert np.all(mv2 == np.argmin(mv.volume))

        mv2 = np.argmax(mv, axis=-1)
        assert np.all(mv2 == np.argmax(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.argmax(mv)
        assert np.all(mv2 == np.argmax(mv.volume))

        # NaN functions
        vol_nan = np.ones(shape)
        vol_nan[..., 1] = np.nan
        mv = MedicalVolume(vol_nan, np.eye(4), headers=headers)

        mv2 = np.nansum(mv, axis=-1)
        assert np.all(mv2 == np.nansum(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nansum(mv)
        assert np.all(mv2 == np.nansum(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.nanmean(mv, axis=-1)
        assert np.all(mv2 == np.nanmean(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanmean(mv)
        assert np.all(mv2 == np.nanmean(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.nanstd(mv, axis=-1)
        assert np.all(mv2 == np.nanstd(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanstd(mv)
        assert np.all(mv2 == np.nanstd(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.nan_to_num(mv)
        assert np.unique(mv2.volume).tolist() == [0, 1]
        mv2 = np.nan_to_num(mv, copy=False)
        assert id(mv2) == id(mv)

        mv2 = np.nanmin(mv, axis=-1)
        assert np.all(mv2 == np.nanmin(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanmin(mv)
        assert np.all(mv2 == np.nanmin(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.nanmax(mv, axis=-1)
        assert np.all(mv2 == np.nanmax(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanmax(mv)
        assert np.all(mv2 == np.nanmax(mv.volume))
        assert np.isscalar(mv2)

        mv2 = np.nanargmin(mv, axis=-1)
        assert np.all(mv2 == np.nanargmin(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanargmin(mv)
        assert np.all(mv2 == np.nanargmin(mv.volume))

        mv2 = np.nanargmax(mv, axis=-1)
        assert np.all(mv2 == np.nanargmax(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.nanargmax(mv)
        assert np.all(mv2 == np.nanargmax(mv.volume))

        # Round
        shape = (10, 20, 30, 2)
        affine = np.concatenate([np.random.rand(3, 4), [[0, 0, 0, 1]]], axis=0)
        mv = MedicalVolume(np.random.rand(*shape), affine, headers=headers)

        mv2 = mv.round()
        assert np.allclose(mv2.affine, affine)
        assert np.unique(mv2.volume).tolist() == [0, 1]

        mv2 = mv.round(affine=True)
        assert np.unique(mv2.affine).tolist() == [0, 1]
        assert np.unique(mv2.volume).tolist() == [0, 1]

        # Clip
        shape = (10, 20, 30)
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4))

        mv2 = np.clip(mv, 0.4, 0.6)
        assert np.all((mv2.volume >= 0.4) & (mv2.volume <= 0.6))

        mv_lower = MedicalVolume(np.ones(mv.shape) * 0.4, mv.affine)
        mv_upper = MedicalVolume(np.ones(mv.shape) * 0.6, mv.affine)
        mv2 = np.clip(mv, mv_lower, mv_upper)
        assert np.all((mv2.volume >= 0.4) & (mv2.volume <= 0.6))

        # Array like
        shape = (10, 20, 30)
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4))

        mv2 = np.zeros_like(mv)
        assert np.all(mv2.volume == 0)

        mv2 = np.ones_like(mv)
        assert np.all(mv2.volume == 1)

        # Shares memory
        shape = (10, 20, 30, 2)
        headers = np.stack(
            [
                ututils.build_dummy_headers(shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4), headers=headers)
        mv2 = MedicalVolume(mv.A, affine=mv.affine, headers=mv.headers())
        assert np.shares_memory(mv, mv)
        assert np.shares_memory(mv, mv2)

    def test_numpy_shaping(self):
        """Test numpy shaping functions (stack, concatenate, etc.)."""
        shape = (10, 20, 30, 2)
        headers = np.stack(
            [
                ututils.build_dummy_headers(shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        vol = np.ones(shape)
        mv_a = MedicalVolume(vol, np.eye(4), headers=headers)
        mv_b = MedicalVolume(vol * 2, np.eye(4), headers=headers)
        mv_c = MedicalVolume(vol * 3, np.eye(4), headers=headers)

        assert mv_a.shape == vol.shape
        assert mv_a.ndim == vol.ndim

        # Stack
        mv2 = np.stack([mv_a, mv_b, mv_c], axis=-1)
        assert mv2.shape == (10, 20, 30, 2, 3)
        assert mv2.headers() is not None
        assert np.all(mv2.volume == np.stack([mv_a.volume, mv_b.volume, mv_c.volume], axis=-1))
        mv2 = np.stack([mv_a, mv_b, mv_c], axis=-2)
        assert mv2.shape == (10, 20, 30, 3, 2)
        with self.assertRaises(ValueError):
            mv2 = np.stack([mv_a, mv_b, mv_c], axis=0)
        with self.assertRaises(TypeError):
            mv2 = np.stack([mv_a, mv_b, mv_c], axis=(-1,))

        # Expand dims
        mv2 = np.expand_dims(mv_a, (-2, -3))
        assert mv2.shape == (10, 20, 30, 1, 1, 2)
        mv2 = np.expand_dims(mv_a, -1)
        assert mv2.shape == (10, 20, 30, 2, 1)
        with self.assertRaises(ValueError):
            mv2 = np.expand_dims(mv_a, 0)

        # Squeeze
        mv_d = mv_a[..., :1]
        assert mv_d.shape == (10, 20, 30, 1)
        assert np.squeeze(mv_d).shape == (10, 20, 30)
        assert np.squeeze(mv_d, axis=-1).shape == (10, 20, 30)
        assert np.squeeze(mv_d, axis=3).shape == (10, 20, 30)
        assert np.squeeze(mv_d, axis=3).shape == (10, 20, 30)

        mv_d = mv_a[:1, :, :, :1]
        assert np.squeeze(mv_d).shape == (1, 20, 30)
        with self.assertRaises(ValueError):
            np.squeeze(mv_d, axis=0)

        # Concatenate
        with self.assertRaises(ValueError):
            np.concatenate([mv_a, mv_b], axis=0)
        with self.assertRaises(TypeError):
            np.concatenate([mv_a, mv_b], axis="-1")

        mv2 = np.concatenate([mv_a, mv_b], axis=-1)
        assert np.all(mv2.volume == np.concatenate([mv_a.volume, mv_b.volume], axis=-1))
        assert mv2.headers().shape == (1, 1, 30, 4)

        affine = np.eye(4)
        affine[:3, 3] += [10, 0, 0]
        mv_d = MedicalVolume(vol * 2, affine, headers=headers)
        with self.assertWarns(UserWarning):
            mv2 = np.concatenate([mv_a, mv_d], axis=0)
        assert mv2.headers() is None
        assert np.all(mv2.volume == np.concatenate([mv_a.volume, mv_d.volume], axis=0))

        affine = np.eye(4)
        affine[:3, 3] += [0, 0, 30]
        mv_d = MedicalVolume(vol * 2, affine, headers=headers)
        mv2 = np.concatenate([mv_a, mv_d], axis=-2)
        assert np.all(mv2.volume == np.concatenate([mv_a.volume, mv_d.volume], axis=-2))
        assert mv2.headers().shape == (1, 1, 60, 2)

        affine = np.eye(4)
        affine[:3, 3] += [0, 0, 30]
        affine[:, 0] *= 0.5
        mv_d = MedicalVolume(vol * 2, affine, headers=headers)
        with self.assertRaises(ValueError):
            mv2 = np.concatenate([mv_a, mv_d], axis=-2)

        affine = np.eye(4)
        affine[:3, 3] += [0, 0, 30]
        mv_d = MedicalVolume(vol * 2, affine, headers=None)
        mv2 = np.concatenate([mv_a, mv_d], axis=-2)
        assert np.all(mv2.volume == np.concatenate([mv_a.volume, mv_d.volume], axis=-2))
        assert mv2.headers() is None

    def test_hdf5(self):
        shape = (10, 20, 30)
        volume = np.reshape(list(range(np.product(shape))), shape)
        hdf5_file = os.path.join(self._TEMP_PATH, "unittest.h5")

        with h5py.File(hdf5_file, "w") as f:
            f.create_dataset("volume", data=volume)
        f = h5py.File(hdf5_file, "r")

        mv = MedicalVolume(f["volume"], np.eye(4))
        assert mv.device == Device("cpu")
        assert mv.dtype == f["volume"].dtype

        mv2 = mv[:, :, :1]
        assert np.all(mv2.volume == volume[:, :, :1])
        assert mv2.device == Device("cpu")
        assert mv2.dtype == volume.dtype

    def test_comparison(self):
        mv1 = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = MedicalVolume(2 * np.ones((10, 20, 30)), self._AFFINE)

        assert np.all((mv1 == mv1.clone()).volume)
        assert np.all((mv1 != mv2).volume)
        assert np.all((mv1 < mv2).volume)
        assert np.all((mv1 <= mv2).volume)
        assert np.all((mv2 > mv1).volume)
        assert np.all((mv2 >= mv1).volume)

    def test_slice(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        with self.assertRaises(IndexError):
            mv[4]
        mv_slice = mv[4:5]
        assert mv_slice.shape == (1, 20, 30)

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv[:5, ...] = 2
        assert np.all(mv._volume[:5, ...] == 2) & np.all(mv._volume[5:, ...] == 1)
        assert np.all(mv[:5, ...].volume == 2)

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv2 = mv[:5, ...].clone()
        mv2 += 2
        mv[:5, ...] = mv2
        assert np.all(mv._volume[:5, ...] == 3) & np.all(mv._volume[5:, ...] == 1)
        assert np.all(mv[:5, ...].volume == 3)

    def test_slice_with_headers(self):
        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        headers = np.stack(
            [
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        mv = MedicalVolume(vol, self._AFFINE, headers=headers)

        mv2 = mv[..., 0]
        assert mv2._headers.shape == (1, 1, 30)
        for h in mv2.headers(flatten=True):
            assert h["EchoTime"].value == 2

        mv2 = mv[..., 1]
        assert mv2._headers.shape == (1, 1, 30)
        for h in mv2.headers(flatten=True):
            assert h["EchoTime"].value == 10

        mv2 = mv[:10, :5, 8:10, :1]
        assert mv2._headers.shape == (1, 1, 2, 1)

        mv2 = mv[:10]
        assert mv2._headers.shape == (1, 1, 30, 2)
        mv2 = mv[:, :10]
        assert mv2._headers.shape == (1, 1, 30, 2)

        mv2 = mv[..., 0:1]
        assert mv2._headers.shape == (1, 1, 30, 1)

        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        headers = ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2})[..., np.newaxis]
        mv = MedicalVolume(vol, self._AFFINE, headers=headers)
        mv1 = mv[..., 0]
        mv2 = mv[..., 1]
        assert mv1._headers.shape == (1, 1, 30)
        assert mv2._headers.shape == (1, 1, 30)
        for h1, h2 in zip(mv1.headers(flatten=True), mv2.headers(flatten=True)):
            assert id(h1) == id(h2)

    def test_4d(self):
        vol = np.stack([np.ones((10, 20, 30)), 2 * np.ones((10, 20, 30))], axis=-1)
        mv = MedicalVolume(vol, self._AFFINE)
        assert mv.orientation == ("SI", "AP", "LR")
        assert mv.shape == (10, 20, 30, 2)

        assert np.all(mv[..., 0].volume == 1)
        assert np.all(mv[..., 1].volume == 2)

        ornt = ("AP", "IS", "RL")
        mv2 = mv.reformat(ornt)
        assert mv2.orientation == ornt
        assert mv2.shape == (20, 10, 30, 2)

        mv2 = mv.reformat(ornt).reformat(mv.orientation)
        assert mv2.is_identical(mv)

        fp = os.path.join(self._TEMP_PATH, "test_4d.nii.gz")
        mv.save_volume(fp)
        mv2 = NiftiReader().load(fp)
        assert mv2.is_identical(mv)

    def test_device(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        assert mv.to(Device(-1)) == mv
        assert mv.cpu() == mv

    @ututils.requires_packages("cupy")
    def test_device_gpu(self):
        import cupy as cp

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
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

    def test_array_cpu(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)

        data = np.asarray(mv)
        assert np.shares_memory(data, mv.volume)

    @ututils.requires_packages("cupy")
    def test_array_gpu(self):
        import cupy as cp

        mv = MedicalVolume(np.ones((10, 20, 30)), self._AFFINE)
        mv_gpu = mv.to(Device(0))
        data = cp.asarray(mv_gpu)
        assert cp.shares_memory(data, mv_gpu.volume)

    def test_dtype(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        assert mv.volume.dtype == vol.dtype

        mv2 = mv.astype("int32")
        assert id(mv) == id(mv2)
        assert mv2.volume.dtype == np.int32

    def test_repr(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        assert mv.__repr__() is not None

    def test_set_volume(self):
        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        mv.volume += 2
        assert np.all(mv.volume == 3)

    @ututils.requires_packages("torch")
    def test_to_torch(self):
        import torch

        vol = np.ones((10, 20, 30))
        mv = MedicalVolume(vol, self._AFFINE)

        tensor = mv.to_torch()
        assert torch.all(tensor == torch.from_numpy(vol))
        assert tensor.shape == mv.shape

        tensor = mv.to_torch(requires_grad=True, contiguous=True)
        assert tensor.is_contiguous()
        assert tensor.requires_grad
        assert torch.all(tensor == torch.from_numpy(vol))
        assert tensor.shape == mv.shape

        vol = np.ones((10, 20, 30), np.complex)
        mv = MedicalVolume(vol, self._AFFINE)

        tensor = mv.to_torch()
        assert tensor.dtype == torch.complex128
        assert tensor.shape == mv.shape

        tensor = mv.to_torch(view_as_real=True)
        assert tensor.shape == mv.shape + (2,)

    @ututils.requires_packages("torch")
    def test_from_torch(self):
        import torch

        tensor = torch.ones(10, 20, 30)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert np.all(tensor.numpy() == mv.A)

        tensor = torch.ones(10, 20, 30)
        mv = MedicalVolume.from_torch(tensor, torch.from_numpy(self._AFFINE))
        assert np.all(tensor.numpy() == mv.A)
        assert isinstance(mv.affine, np.ndarray)

        tensor = torch.ones(10, 20, 30, dtype=torch.complex64)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert mv.dtype == np.complex64

        tensor = torch.ones(10, 20, 30, dtype=torch.complex128)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE)
        assert mv.dtype == np.complex128

        tensor = torch.ones(10, 20, 30, 2, dtype=torch.float32)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)
        assert mv.dtype == np.complex64
        assert mv.shape == tensor.shape[:3]

        tensor = torch.ones(10, 20, 30, 2, dtype=torch.float64)
        mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)
        assert mv.dtype == np.complex128
        assert mv.shape == tensor.shape[:3]

        tensor = torch.ones(10, 20, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE)

        tensor = torch.ones(10, 20, 2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)

        tensor = torch.ones(10, 20, 30, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            mv = MedicalVolume.from_torch(tensor, self._AFFINE, to_complex=True)


if __name__ == "__main__":
    unittest.main()
