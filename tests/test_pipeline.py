import os
import unittest

import numpy as np

import file_constants as fc
import pipeline
from scan_sequences.cube_quant import CubeQuant
from tissues.femoral_cartilage import FemoralCartilage
from utils import io_utils, dicom_utils

DESS_003_DICOM_PATH = '../dicoms/003'
DESS_003_T2_MAP_PATH = '../dicoms/003_t2_map.mat'

SAVE_PATH = '../dicoms/healthy07/data'
DESS_DICOM_PATH = '../dicoms/healthy07/007'
CUBEQUANT_DICOM_PATH = '../dicoms/healthy07/008'
CUBEQUANT_INTERREGISTERED_PATH = '../dicoms/healthy07/data/cube_quant_data/interregistered'

CUBEQUANT_T1_RHO_MAP_PATH = ''

DECIMAL_PRECISION = 1  # (+/- 0.1ms)


class CubeQuantTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # launch dess as initializer for registration (only need to call this once per session)
        super(CubeQuantTest, cls).setUpClass()

        dt = DessTest()
        dess_vargin = dt.get_vargin()
        dess_vargin[pipeline.DICOM_KEY] = DESS_DICOM_PATH
        dess_vargin[pipeline.SAVE_KEY] = SAVE_PATH

        scan = pipeline.handle_qdess(dess_vargin)
        pipeline.save_info(dess_vargin[pipeline.SAVE_KEY], scan)

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

    def test_init_cubequant(self):
        cq = CubeQuant(CUBEQUANT_DICOM_PATH, 'dcm', './')

    def base_interregister(self, mask_path=None):
        # Interregister cubequant files
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


class UtilsTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_simpleitk_io(self):
        # Read and write dicom volumes using simpleitk
        arr, refs_dicom, spacing = dicom_utils.load_dicom(DESS_DICOM_PATH, 'dcm')

        filepath = './dicoms/ex.nii.gz'

        io_utils.save_nifti(filepath, arr, spacing)
        arr2, spacing2 = io_utils.load_nifti(filepath)

        print(spacing)

        assert (arr == arr2).all(), "Saved and loaded array must be the same"
        assert spacing == spacing2, "spacing should agree"

    def test_h5(self):
        filepath = './dicoms/sample_h5_data'
        datas = [{'type': np.zeros([4, 4, 4]), 'type2': np.zeros([4, 4, 4])}]

        for data in datas:
            io_utils.save_h5(filepath, data)
            data2 = io_utils.load_h5(filepath)

            assert len(list(data.keys())) == len(list(data2.keys()))

            for key in data.keys():
                assert (data[key] == data2[key]).all()


class ExportTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_load_modified_mask(self):
        filepath = './dicoms/healthy07/data/fc_modified.nii.gz'
        arr2, spacing2 = io_utils.load_nifti(filepath)

        print(arr2.shape)
        print(spacing2)


class TissuesTest(unittest.TestCase):
    def segment_dess(self):
        dt = DessTest()
        dess_vargin = dt.get_vargin()
        dess_vargin[pipeline.DICOM_KEY] = DESS_DICOM_PATH
        dess_vargin[pipeline.SAVE_KEY] = SAVE_PATH
        dess_vargin[pipeline.T2_KEY] = True

        scan = pipeline.handle_qdess(dess_vargin)
        pipeline.save_info(dess_vargin[pipeline.SAVE_KEY], scan)

    def test_femoral_cartilage(self):

        if not os.path.isdir(SAVE_PATH):
            self.segment_dess()

        vargin = {'dicom': None, 'save': 'dicoms/healthy07/data', 'load': 'dicoms/healthy07/data', 'ext': 'dcm',
                  'gpu': None,
                  'scan': 'tissues', 'fc': True, 't2': 'dicoms/healthy07/data/dess_data/t2.h5', 't1_rho': None,
                  't2_star': None, 'tissues': [FemoralCartilage()]}

        tissues = pipeline.handle_tissues(vargin)

        for tissue in tissues:
            tissue.save_data(vargin[pipeline.SAVE_KEY])


class FileConstantsTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_reinit_variables(self):
        do, oo = fc.DEBUG, fc.NIPYPE_LOGGING

        fc.set_debug()

        dn, on = fc.DEBUG, fc.NIPYPE_LOGGING

        assert do == 0
        assert dn == 1

        assert oo == 'none'
        assert on == 'stream'


if __name__ == '__main__':
    unittest.main()
