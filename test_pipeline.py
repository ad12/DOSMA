import unittest
import keras.backend as K
import numpy as np
import scipy.io as sio

import pipeline
from tissues.femoral_cartilage import FemoralCartilage
from scan_sequences.cube_quant import CubeQuant

DESS_003_DICOM_PATH = './dicoms/003'
DESS_003_T2_MAP_PATH = './dicoms/003_t2_map.mat'

CUBEQUANT_DICOM_PATH = ''
CUBEQUANT_T1_RHO_MAP_PATH = ''

DECIMAL_PRECISION = 1 # (+/- 0.1ms)


class DessTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def get_vargin(self):
        vargin = dict()
        vargin[pipeline.TISSUES_KEY] = [FemoralCartilage()]
        vargin[pipeline.DICOM_KEY] = DESS_003_DICOM_PATH
        vargin[pipeline.SAVE_KEY] = DESS_003_DICOM_PATH
        vargin[pipeline.EXT_KEY] = 'dcm'
        vargin[pipeline.SEGMENTATION_MODEL_KEY] = 'unet2d'
        vargin[pipeline.SEGMENTATION_WEIGHTS_DIR_KEY] = './weights'
        vargin[pipeline.ACTION_KEY] = 'segment'
        vargin[pipeline.SEGMENTATION_BATCH_SIZE_KEY] = 32
        vargin[pipeline.T2_KEY] = False
        return vargin

    def test_batch_size(self):
        """Test if batch size makes a difference on the output"""
        vargin = self.get_vargin()
        scan = pipeline.handle_dess(vargin)
        mask_32 = scan.tissues[0].mask
        K.clear_session()

        vargin = self.get_vargin()
        vargin[pipeline.SEGMENTATION_BATCH_SIZE_KEY] = 16
        scan = pipeline.handle_dess(vargin)
        mask_16 = scan.tissues[0].mask
        K.clear_session()

        vargin = self.get_vargin()
        vargin[pipeline.SEGMENTATION_BATCH_SIZE_KEY] = 4
        scan = pipeline.handle_dess(vargin)
        mask_4 = scan.tissues[0].mask
        K.clear_session()

        assert (mask_32 == mask_4).all()
        assert (mask_32 == mask_16).all()

    def test_t2_map(self):
        # Load ground truth t2 map
        vargin = self.get_vargin()
        vargin[pipeline.T2_KEY] = True

        scan = pipeline.handle_dess(vargin)

        mat_t2_map = sio.loadmat(DESS_003_T2_MAP_PATH)
        mat_t2_map = mat_t2_map['t2map']

        # calculate t2 map
        py_t2_map = scan.t2map

        # need to convert all np.nan values to 0 before comparing
        # np.nan does not equal np.nan, so we need the same values to compare
        mat_t2_map = np.nan_to_num(mat_t2_map)
        #py_t2_map = np.nan_to_num(py_t2_map)

        # Round to the nearest 1000th (0.001)
        mat_t2_map = np.round(mat_t2_map, decimals=DECIMAL_PRECISION)
        py_t2_map = np.round(py_t2_map, decimals=DECIMAL_PRECISION)

        assert((mat_t2_map == py_t2_map).all())

    def test_loading_data(self):
        vargin = self.get_vargin()
        vargin[pipeline.T2_KEY] = True

        scan = pipeline.handle_dess(vargin)
        mask = scan.tissues[0].mask
        t2_map = scan.t2map

        pipeline.save_info(vargin[pipeline.SAVE_KEY], scan)


        fc = FemoralCartilage()
        fc.load_data(vargin[pipeline.SAVE_KEY])

        mask2 = fc.mask

        assert((mask == mask2).all())


class CubeQuantTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def get_vargin(self):
        vargin = dict()
        vargin[pipeline.TISSUES_KEY] = [FemoralCartilage()]
        vargin[pipeline.DICOM_KEY] = CUBEQUANT_DICOM_PATH
        vargin[pipeline.SAVE_KEY] = CUBEQUANT_DICOM_PATH
        vargin[pipeline.EXT_KEY] = 'dcm'
        vargin[pipeline.T1_RHO_Key] = False
        return vargin

    def test_registration(self):
        vargin = self.get_vargin()
        scan = pipeline.handle_cubequant(vargin)


if __name__ == '__main__':
    #unittest.main()
    import os

    print(os.environ['DYLD_LIBRARY_PATH'])
    cq = CubeQuant('./dicoms/healthy07/008', 'dcm', './')
