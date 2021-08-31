import os
import unittest

import numpy as np

from dosma.core.io.format_io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import T2

from .. import util as ututils


class TestT2(unittest.TestCase):
    _AFFINE = np.asarray(
        [
            [0.0, 0.0, 0.8, -171.41],
            [0.0, -0.3125, 0.0, 96.0154],
            [-0.3125, 0.0, 0.0, 47.0233],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # ('SI', 'AP', 'LR')

    def test_basic(self):
        vol = np.zeros((10, 10, 10))
        mv = MedicalVolume(vol, self._AFFINE)
        with self.assertRaises(TypeError):
            _ = T2(vol)

        qv = T2(mv)
        qv.add_additional_volume("r2", mv + 1)
        assert np.all(qv.additional_volumes["r2"] == mv + 1)

    def test_save_load_data(self):
        out_dir = os.path.join(ututils.TEMP_PATH, "quant_val_save")
        vol = np.zeros((10, 10, 10))
        mv = MedicalVolume(vol, self._AFFINE)

        qv = T2(mv)
        qv.add_additional_volume("r2", mv + 1)

        qv.save_data(out_dir, ImageDataFormat.nifti)
        assert os.path.isfile(os.path.join(out_dir, "t2", "t2.nii.gz"))
        assert os.path.isfile(os.path.join(out_dir, "t2", "t2-r2.nii.gz"))

        with self.assertWarns(UserWarning):
            qv.save_data(out_dir, ImageDataFormat.dicom)

        qv2 = T2()
        qv2.load_data(out_dir)
        assert qv2.volumetric_map.is_identical(qv.volumetric_map)

    def test_to_metrics(self):
        vol = np.zeros((10, 10, 10))
        vol[:5, :5, :] = 1
        vol[:5, 5:, :] = 2
        vol[5:, :5, :] = 3
        vol[5:, 5:, :] = 4
        mask = MedicalVolume(vol, self._AFFINE)

        qv_vol = 10 * vol
        t2 = T2(MedicalVolume(qv_vol, self._AFFINE))

        metrics = t2.to_metrics()
        assert np.allclose(metrics["Mean"], [25]), metrics["Mean"].tolist()
        assert np.allclose(metrics["Median"], [25]), metrics["Median"].tolist()
        assert np.allclose(metrics["Std"], [11.180339887498949]), metrics["Std"].tolist()

        metrics = t2.to_metrics(mask)
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert np.allclose(metrics["Mean"], [10, 20, 30, 40, 25]), metrics["Mean"].tolist()
        assert np.allclose(metrics["Median"], [10, 20, 30, 40, 25]), metrics["Median"].tolist()
        assert np.allclose(metrics["Std"], [0, 0, 0, 0, 11.180339887498949]), metrics[
            "Std"
        ].tolist()
        assert metrics["# Voxels"].tolist() == [250, 250, 250, 250, 1000], metrics[
            "# Voxels"
        ].tolist()

        metrics = t2.to_metrics(mask, labels={1: "A", 2: "B", 3: "C", 4: "D"})
        assert metrics["Category"].tolist() == ["A", "B", "C", "D", "total"]

        metrics = t2.to_metrics(mask, bounds=(0, 30))
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [250, 250, 250, 0, 750], metrics["# Voxels"].tolist()

        metrics = t2.to_metrics(mask, bounds=(10, 40), closed="neither")
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [0, 250, 250, 0, 500], metrics["# Voxels"].tolist()

        metrics = t2.to_metrics(mask, bounds=(10, 40), closed="both")
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [250, 250, 250, 250, 1000], metrics[
            "# Voxels"
        ].tolist()

        metrics = t2.to_metrics(mask, bounds=(10, 40), closed="left")
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [250, 250, 250, 0, 750], metrics["# Voxels"].tolist()

        vol_zero = np.copy(vol)
        vol_zero[:5, :5, :] = 0
        mask_zero = MedicalVolume(vol_zero, self._AFFINE)
        metrics = t2.to_metrics(mask_zero)
        assert metrics["Category"].tolist() == ["label_2", "label_3", "label_4", "total"], metrics[
            "Category"
        ].tolist()
        assert np.allclose(metrics["Mean"], [20, 30, 40, 30]), metrics["Mean"].tolist()
        assert np.allclose(metrics["Std"], [0, 0, 0, 8.164966]), metrics["Std"].tolist()

        vol_nan = np.copy(vol)
        vol_nan[:5, :5, :] = np.nan
        mask_nan = MedicalVolume(vol_zero, self._AFFINE)
        metrics = t2.to_metrics(mask_nan)
        assert metrics["Category"].tolist() == ["label_2", "label_3", "label_4", "total"], metrics[
            "Category"
        ].tolist()
        assert np.allclose(metrics["Mean"], [20, 30, 40, 30]), metrics["Mean"].tolist()
        assert np.allclose(metrics["Std"], [0, 0, 0, 8.164966]), metrics["Std"].tolist()

        qv_vol_zero = np.copy(qv_vol)
        qv_vol_zero[:5, :5, :] = 0
        t2_zero = T2(MedicalVolume(qv_vol_zero, self._AFFINE))
        metrics = t2_zero.to_metrics(mask, bounds=(0, np.inf))
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [0, 250, 250, 250, 750], metrics["# Voxels"].tolist()

        qv_vol_nan = np.copy(qv_vol)
        qv_vol_nan[:5, :5, :] = np.nan
        t2_nan = T2(MedicalVolume(qv_vol_nan, self._AFFINE))
        metrics = t2_nan.to_metrics(mask)
        assert metrics["Category"].tolist() == [
            "label_1",
            "label_2",
            "label_3",
            "label_4",
            "total",
        ], metrics["Category"].tolist()
        assert metrics["# Voxels"].tolist() == [0, 250, 250, 250, 750], metrics["# Voxels"].tolist()


if __name__ == "__main__":
    unittest.main()
