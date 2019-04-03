import os
import unittest

import numpy as np

from scan_sequences.cube_quant import CubeQuant
from tissues.femoral_cartilage import FemoralCartilage
from utils import io_utils, dicom_utils

CUBEQUANT_DICOM_PATH = '../dicoms/healthy07/008'
CUBEQUANT_INTERREGISTERED_PATH = '../dicoms/healthy07/data/cube_quant_data/interregistered'

CUBEQUANT_T1_RHO_MAP_PATH = ''

DECIMAL_PRECISION = 1  # (+/- 0.1ms)


class CubeQuantTest(unittest.TestCase):

    def setUp(self):
        print("Testing: ", self._testMethodName)

    def get_vargin(self):
        vargin = dict()
        vargin[pipeline.TISSUES_KEY] = [FemoralCartilage()]
        vargin[pipeline.DICOM_KEY] = CUBEQUANT_DICOM_PATH
        vargin[pipeline.SAVE_KEY] = SAVE_PATH
        vargin[pipeline.EXT_KEY] = 'dcm'
        vargin[pipeline.T1_RHO_Key] = False
        vargin[pipeline.INTERREGISTERED_FILES_DIR_KEY] = None
        vargin[pipeline.ACTION_KEY] = None
        vargin[pipeline.TARGET_SCAN_KEY] = None
        vargin[pipeline.TARGET_MASK_KEY] = None
        vargin[pipeline.LOAD_KEY] = None

        return vargin

    def base_interregister(self, mask_path=None):
        # Interregister cubequant files
        cq = CubeQuant()
        vargin = self.get_vargin()
        vargin[pipeline.ACTION_KEY] = 'interregister'
        vargin[pipeline.TARGET_SCAN_KEY] = os.path.join(SAVE_PATH, 'dess_data', 'dess-interregister.nii.gz')
        vargin[pipeline.TARGET_MASK_KEY] = mask_path

        scan = pipeline.handle_cubequant(vargin)
        pipeline.save_info(vargin[pipeline.SAVE_KEY], scan)

        return scan

    def test_interregister_no_mask(self):
        self.base_interregister()

    def test_interregister_mask(self):
        # Interregister cubequant files using mask
        # assume dess segmentation exists
        self.base_interregister('./dicoms/healthy07/data/fc.nii.gz')

    def test_load_interregister(self):
        # make sure subvolumes are the same
        base_scan = self.base_interregister()

        vargin = self.get_vargin()
        vargin[pipeline.INTERREGISTERED_FILES_DIR_KEY] = CUBEQUANT_INTERREGISTERED_PATH
        load_scan = pipeline.handle_cubequant(vargin)

        # subvolume lengths must be the same
        assert len(base_scan.subvolumes) == len(load_scan.subvolumes), \
            'Length mismatch - %d subvolumes (base), %d subvolumes (load)' % (len(base_scan.subvolumes),
                                                                              len(load_scan.subvolumes))
        assert len(load_scan.subvolumes) == 4

        for key in base_scan.subvolumes.keys():
            base_subvolume = base_scan.subvolumes[key]
            load_subvolume = load_scan.subvolumes[key]
            assert np.array_equal(base_subvolume, load_subvolume), "Failed key: %s" % str(key)

    def test_t1_rho_map(self):
        vargin = self.get_vargin()
        vargin[pipeline.INTERREGISTERED_FILES_DIR_KEY] = CUBEQUANT_INTERREGISTERED_PATH
        vargin[pipeline.T1_RHO_Key] = True
        vargin[pipeline.LOAD_KEY] = SAVE_PATH

        scan = pipeline.handle_cubequant(vargin)
        pipeline.save_info(vargin[pipeline.SAVE_KEY], scan)

        assert scan.t1rho_map is not None

    def test_dicom_negative(self):
        """Load dicoms and see if any negative values exist"""
        arr, _, _ = dicom_utils.load_dicom(CUBEQUANT_DICOM_PATH, 'dcm')

        assert np.sum(arr < 0) == 0, "No dicom values should be negative"

    def test_baseline_raw_negative(self):
        base_filepath = './dicoms/healthy07/cubequant_elastix_baseline/raw/%03d.nii.gz'
        spin_lock_times = [1, 10, 30, 60]

        for sl in spin_lock_times:
            filepath = base_filepath % sl
            arr, _ = io_utils.load_nifti(filepath)

            assert np.sum(arr < 0) == 0, "Failed %03d: no values should be negative" % sl
