import os
import unittest

import numpy as np

from data_io import ImageDataFormat, NiftiReader
from scan_sequences import CubeQuant, QDess
from tissues.femoral_cartilage import FemoralCartilage
from .. import util

DECIMAL_PRECISION = 1  # (+/- 0.1ms)

# target mask path used to register Cubequant volume to qDESS volume
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(CubeQuant.NAME), 'misc/fc.nii.gz')


class CubeQuantTest(util.ScanTest):
    SCAN_TYPE = CubeQuant

    def test_interregister_no_mask(self):
        """Register Cubequant scan to qDESS scan without a target mask"""
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)

        # Register to first echo of QDess without a mask
        qdess_echo1_path = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
        scan.interregister(target_path=qdess_echo1_path)

    def test_interregister_mask(self):
        """Register Cubequant scan to qDESS scan with a target mask (mask for femoral cartilage)"""
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)

        qdess_echo1_path = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
        scan.interregister(target_path=qdess_echo1_path, target_mask_path=TARGET_MASK_PATH)

    def test_t1_rho_map(self):
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        qdess_echo1_path = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
        scan.interregister(target_path=qdess_echo1_path, target_mask_path=TARGET_MASK_PATH)

        # run analysis with femoral cartilage, without mask
        tissue = FemoralCartilage()
        map1 = scan.generate_t1_rho_map(tissue, TARGET_MASK_PATH)
        assert map1 is not None, "map should not be None"

        # add mask to femoral cartilage and run
        nr = NiftiReader()
        tissue.set_mask(nr.load(TARGET_MASK_PATH))
        map2 = scan.generate_t1_rho_map(tissue)
        assert map2 is not None, "map should not be None"

        # map1 and map2 should be identical
        assert(map1.volumetric_map.is_identical(map2.volumetric_map))


if __name__ == '__main__':
    unittest.main()