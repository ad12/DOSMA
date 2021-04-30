import os
import unittest

import numpy as np

from dosma.core.io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.scan_sequences.mri import Cones, QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage
from dosma.utils import io_utils

from ... import util

# target mask path used to register Cubequant volume to qDESS volume
if util.is_data_available():
    QDESS_ECHO1_PATH = util.get_read_paths(
        util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti
    )[0]
    TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(Cones.NAME), "misc/fc.nii.gz")
else:
    QDESS_ECHO1_PATH = None
    TARGET_MASK_PATH = None


class ConesTest(util.ScanTest):
    SCAN_TYPE = Cones

    def _generate_mock_data(self, shape=None, ts=None):
        """Generates multi-echo mock data for Cones sequence.

        For some echo time :math:`t`, the data can be modeled as:

            :math:`y=a * \\exp(-t/t2star)`

        Args:
            ts: Echo times.
            t2star (ndarray): The t2star times for each voxel
            a (ndarray or int): The multiplicative constant.
        """
        if shape is None:
            shape = (10, 10, 10)
        if ts is None:
            ts = [0.5, 2.0, 3.0, 8.0]

        a = 1.0
        t2star = np.random.rand(*shape) * 80 + 0.1
        _, ys, _, _ = util.generate_monoexp_data(shape=shape, x=ts, a=a, b=-1 / t2star)

        for idx, (y, t) in enumerate(zip(ys, ts)):
            y.set_metadata("EchoTime", t, force=True)
            y.set_metadata("EchoNumber", idx + 1, force=True)

        return ys, ts, a, t2star

    @unittest.skipIf(
        not util.is_data_available() or not util.is_elastix_available(),
        "unittest data or elastix is not available",
    )
    def test_interregister(self):
        """Test Cones interregistration."""
        # Register to first echo of QDess without a mask.
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH)

        # Register to first echo of QDess with a mask.
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

    def test_save_load(self):
        ys, _, _, _ = self._generate_mock_data()
        scan = Cones(ys)

        save_dir = os.path.join(self.data_dirpath, "test-save")
        save_path = scan.save(save_dir, save_custom=True, image_data_format=ImageDataFormat.nifti)
        assert set(os.listdir(save_dir)) == {"volumes", f"{scan.NAME}.data"}

        scan2 = Cones.load(save_dir)
        for v1, v2 in zip(scan.volumes, scan2.volumes):
            assert v1.is_identical(v2)

        scan2 = Cones.load(save_path)
        for v1, v2 in zip(scan.volumes, scan2.volumes):
            assert v1.is_identical(v2)

        scan2 = Cones.from_dict(io_utils.load_pik(save_path))
        for v1, v2 in zip(scan.volumes, scan2.volumes):
            assert v1.is_identical(v2)

    def test_t2_star_map(self):
        ys, _, _, _ = self._generate_mock_data()
        scan = Cones(ys)

        # No mask
        tissue = FemoralCartilage()
        map1 = scan.generate_t2_star_map(tissue, num_workers=util.num_workers())
        assert map1 is not None, "map should not be None"

        mask = MedicalVolume(np.ones(ys[0].shape), np.eye(4))

        # Use a mask
        tissue.set_mask(mask)
        map2 = scan.generate_t2_star_map(tissue, num_workers=util.num_workers())
        assert map2 is not None, "map should not be None"

        # map1 and map2 should be identical
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

    @unittest.skipIf(
        not util.is_data_available() or not util.is_elastix_available(),
        "unittest data or elastix is not available",
    )
    def test_cmd_line(self):
        # Generate segmentation mask for femoral cartilage via command line.
        cmdline_str = "--d {} --s {} cones --fc interregister --tp {} --tm {}".format(
            self.dicom_dirpath, self.data_dirpath, QDESS_ECHO1_PATH, TARGET_MASK_PATH
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2star map for femoral cartilage, tibial cartilage,
        # and meniscus via command line.
        cmdline_str = "--l {} cones --fc t2_star --mask_path {}".format(
            self.data_dirpath, TARGET_MASK_PATH
        )
        self.__cmd_line_helper__(cmdline_str)


if __name__ == "__main__":
    unittest.main()
