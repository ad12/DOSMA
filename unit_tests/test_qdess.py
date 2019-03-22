import os
import unittest

import numpy as np
import scipy.io as sio

from models.get_model import get_model
from scan_sequences.qdess import QDess
from tissues.femoral_cartilage import FemoralCartilage

SAMPLE_qDESS_DICOM_PATH = '../dicoms/qdess-example'
SAMPLE_qDESS_EXPECTED_T2_MAP_PATH = '../dicoms/qdess-example-t2map.mat'

SEGMENTATION_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), '../weights')
SEGMENTATION_MODEL = 'unet2d'

DECIMAL_PRECISION = 1  # (+/- 0.1ms)


class DessTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_batch_size(self):
        """Test if batch size makes a difference on the segmentation output"""

        batch_sizes = [32, 16, 4]
        outputs = []
        for batch_size in batch_sizes:
            scan = QDess(dicom_path=SAMPLE_qDESS_DICOM_PATH)
            tissue = FemoralCartilage()
            tissue.find_weights(SEGMENTATION_WEIGHTS_FOLDER)
            dims = scan.get_dimensions()
            input_shape = (dims[0], dims[1], 1)
            model = get_model(SEGMENTATION_MODEL,
                              input_shape=input_shape,
                              weights_path=tissue.weights_filepath)
            model.batch_size = batch_size
            scan.segment(model, tissue)

            curr_mask = tissue.get_mask()

            # check the results are the same between all masks
            for i in range(len(outputs)):
                assert curr_mask.is_identical(outputs[i]), \
                    "Mismatch in segmentations between batch sizes %d and %d" % (batch_sizes[i], batch_size)

            outputs.append(curr_mask)

    def test_t2_map(self):
        # Load ground truth t2 map
        scan = QDess(dicom_path=SAMPLE_qDESS_DICOM_PATH)
        tissue = FemoralCartilage()
        scan.generate_t2_map(tissue)

        mat_t2_map = sio.loadmat(SAMPLE_qDESS_EXPECTED_T2_MAP_PATH)
        mat_t2_map = mat_t2_map['t2map']

        # calculate t2 map
        py_t2_map = tissue.quantitative_values[0].volumetric_map.volume

        # need to convert all np.nan values to 0 before comparing
        # np.nan does not equal np.nan, so we need the same values to compare
        mat_t2_map = np.nan_to_num(mat_t2_map)
        # py_t2_map = np.nan_to_num(py_t2_map)

        # Round to the nearest 1000th (0.001)
        mat_t2_map = np.round(mat_t2_map, decimals=DECIMAL_PRECISION)
        py_t2_map = np.round(py_t2_map, decimals=DECIMAL_PRECISION)

        assert (mat_t2_map == py_t2_map).all(), \
            "T2map mismatch - expected T2map (MATLAB) does not match computed T2 map"
