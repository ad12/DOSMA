"""Tests for numpy routines and ufunc on MedicalVolumes."""
import unittest

import numpy as np

from dosma.core.med_volume import MedicalVolume

from .. import util as ututils


class TestNumpyRoutinesMedicalVolume(unittest.TestCase):
    _AFFINE = np.asarray(
        [
            [0.0, 0.0, 0.8, -171.41],
            [0.0, -0.3125, 0.0, 96.0154],
            [-0.3125, 0.0, 0.0, 47.0233],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # ('SI', 'AP', 'LR')

    def _build_volume(self, shape_or_vol=(10, 20, 30, 2), affine=None):
        if isinstance(shape_or_vol, np.ndarray):
            vol = shape_or_vol
            shape = shape_or_vol.shape
        elif isinstance(shape_or_vol, MedicalVolume):
            vol = shape_or_vol.A
            shape = shape_or_vol.shape
        else:
            shape = shape_or_vol
            vol = np.random.rand(*shape)

        if len(shape) != 4:
            raise ValueError(f"shape must correspond to 4D volume. Got {shape}")

        headers = np.stack(
            [
                ututils.build_dummy_headers(shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        if affine is None:
            affine = np.eye(4)
        return MedicalVolume(vol, affine, headers=headers)

    def test_where(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        mv[np.where(mv == 1)] = 5
        assert np.all(mv == 5)
        assert not np.any(mv == 1)
        assert all(mv == 5)
        assert not any(mv == 5)

    def test_exp(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        assert np.all(np.exp(mv.volume) == np.exp(mv).volume)

    def test_ndarray_arithmetic(self):
        mv = MedicalVolume(np.ones((10, 20, 30)), np.eye(4))
        mv2 = mv + np.ones(mv.shape)
        assert all(mv2 == 2)

    def test_reduce_funcs(self):
        shape = (10, 20, 30, 2)
        mv = self._build_volume(shape)

        mv2 = np.add.reduce(mv, -1)
        assert np.all(mv2 == np.add.reduce(mv.volume, axis=-1))
        assert mv2.shape == mv.shape[:3]
        mv2 = np.add.reduce(mv, axis=None)
        assert np.all(mv2 == np.add.reduce(mv.volume, axis=None))
        assert np.isscalar(mv2)

    def test_stats_funcs(self):
        shape = (10, 20, 30, 2)
        mv = self._build_volume(shape)

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

    def test_nan_funcs(self):
        shape = (10, 20, 30, 2)
        mv = self._build_volume(shape)
        headers = mv.headers

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

    def test_round(self):
        shape = (10, 20, 30, 2)
        affine = np.concatenate([np.random.rand(3, 4), [[0, 0, 0, 1]]], axis=0)
        mv = self._build_volume(shape, affine=affine)

        mv2 = mv.round()
        assert np.allclose(mv2.affine, affine)
        assert np.unique(mv2.volume).tolist() == [0, 1]

        mv2 = mv.round(affine=True)
        assert np.unique(mv2.affine).tolist() == [0, 1]
        assert np.unique(mv2.volume).tolist() == [0, 1]

    def test_clip(self):
        # Clip
        shape = (10, 20, 30)
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4))

        mv2 = np.clip(mv, 0.4, 0.6)
        assert np.all((mv2.volume >= 0.4) & (mv2.volume <= 0.6))

        mv_lower = MedicalVolume(np.ones(mv.shape) * 0.4, mv.affine)
        mv_upper = MedicalVolume(np.ones(mv.shape) * 0.6, mv.affine)
        mv2 = np.clip(mv, mv_lower, mv_upper)
        assert np.all((mv2.volume >= 0.4) & (mv2.volume <= 0.6))

    def test_array_like(self):
        shape = (10, 20, 30)
        mv = MedicalVolume(np.random.rand(*shape), np.eye(4))

        mv2 = np.zeros_like(mv)
        assert np.all(mv2.volume == 0)

        mv2 = np.ones_like(mv)
        assert np.all(mv2.volume == 1)

    def test_shares_memory(self):
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

    def test_stack(self):
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

    def test_expand_dims(self):
        mv = self._build_volume()
        mv2 = np.expand_dims(mv, (-2, -3))
        assert mv2.shape == (10, 20, 30, 1, 1, 2)
        mv2 = np.expand_dims(mv, -1)
        assert mv2.shape == (10, 20, 30, 2, 1)
        with self.assertRaises(ValueError):
            mv2 = np.expand_dims(mv, 0)

    def test_squeeze(self):
        mv = self._build_volume()
        mv2 = mv[..., :1]
        assert mv2.shape == (10, 20, 30, 1)
        assert np.squeeze(mv2).shape == (10, 20, 30)
        assert np.squeeze(mv2, axis=-1).shape == (10, 20, 30)
        assert np.squeeze(mv2, axis=3).shape == (10, 20, 30)
        assert np.squeeze(mv2, axis=3).shape == (10, 20, 30)

        mv2 = mv[:1, :, :, :1]
        assert np.squeeze(mv2).shape == (1, 20, 30)
        with self.assertRaises(ValueError):
            np.squeeze(mv2, axis=0)

    def test_concatenate(self):
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

    def test_pad(self):
        shape = (10, 20, 30, 2)
        affine = np.eye(4)
        mv = self._build_volume(shape, affine=affine)

        # Pad all dims.
        pad_width = (1, 2, 3, 4)
        np_pad_width = ((1,), (2,), (3,), (4,))
        expected_origin = np.asarray([x - p for x, p in zip(mv.scanner_origin, pad_width)])
        expected_arr = np.pad(mv.A, np_pad_width)
        mv2 = np.pad(mv, pad_width)
        assert np.all(mv2.A == expected_arr)
        assert np.all(np.asarray(mv2.scanner_origin) == expected_origin)

        # Pad all dims by 1 on both sides.
        pad_width = 1
        np_pad_width = 1
        expected_origin = np.asarray([x - pad_width for x in mv.scanner_origin])
        expected_arr = np.pad(mv.A, np_pad_width)
        mv2 = np.pad(mv, pad_width)
        assert np.all(mv2.A == expected_arr)
        assert np.all(np.asarray(mv2.scanner_origin) == expected_origin)

        # Pad some dims.
        pad_width = (None, 0, 3, 4)
        np_pad_width = ((0,), (0,), (3,), (4,))
        expected_origin = np.asarray(
            [x - p if isinstance(p, int) else x for x, p in zip(mv.scanner_origin, pad_width)]
        )
        expected_arr = np.pad(mv.A, np_pad_width)
        mv2 = np.pad(mv, pad_width)
        assert np.all(mv2.A == expected_arr)
        assert np.all(np.asarray(mv2.scanner_origin) == expected_origin)

        pad_width = (3, 4)
        eff_pad_width = (0, 0, 3, 4)
        np_pad_width = ((0,), (0,), (3,), (4,))
        expected_origin = np.asarray(
            [x - p if isinstance(p, int) else x for x, p in zip(mv.scanner_origin, eff_pad_width)]
        )
        expected_arr = np.pad(mv.A, np_pad_width)
        mv2 = np.pad(mv, pad_width)
        assert np.all(mv2.A == expected_arr)
        assert np.all(np.asarray(mv2.scanner_origin) == expected_origin)
