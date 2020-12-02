import os
import unittest

from dosma.models.util import get_model
from dosma.scan_sequences.qdess import QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage
from .. import util

SEGMENTATION_WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), '../../weights/iwoai-2019-t6-normalized')
SEGMENTATION_MODEL = "iwoai-2019-t6-normalized"


class QDessTest(util.ScanTest):
    SCAN_TYPE = QDess

    def test_segmentation_multiclass(self):
        """Test support for multiclass segmentation."""
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        tissue = FemoralCartilage()
        tissue.find_weights(SEGMENTATION_WEIGHTS_FOLDER),
        dims = scan.get_dimensions()
        input_shape = (dims[0], dims[1], 1)
        model = get_model(
            SEGMENTATION_MODEL,
            input_shape=input_shape,
            weights_path=tissue.weights_file_path
        )
        scan.segment(model, tissue, use_rms=True)

    #
    # def test_t2_map(self):
    #     # Load ground truth t2 map
    #     scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
    #     tissue = FemoralCartilage()
    #     scan.generate_t2_map(tissue)
    #
    #     mat_filepath = os.path.join(util.get_expected_data_path(os.path.dirname(self.dicom_dirpath)), tissue.STR_ID,
    #                                 't2map.mat')
    #     mat_t2_map = sio.loadmat(mat_filepath)
    #     mat_t2_map = mat_t2_map['t2map']
    #
    #     # calculate t2 map
    #     py_t2_map = tissue.quantitative_values[0].volumetric_map.volume
    #
    #     # need to convert all np.nan values ato 0 before comparing
    #     # np.nan does not equal np.nan, so we need the same values to compare
    #     mat_t2_map = np.nan_to_num(mat_t2_map)
    #     py_t2_map = np.nan_to_num(py_t2_map)
    #
    #     npt.assert_almost_equal(mat_t2_map, py_t2_map, decimal=util.DECIMAL_PRECISION)

    def test_cmd_line(self):
        # Generate segmentation mask for femoral cartilage via command line.
        cmdline_str = (
            f"--d {self.dicom_dirpath} --s {self.data_dirpath} qdess --fc "
            f"segment --weights_dir {SEGMENTATION_WEIGHTS_FOLDER} "
            f"--model {SEGMENTATION_MODEL} --use_rms"
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2 map for femoral cartilage, tibial cartilage, and meniscus.
        cmdline_str = (
            f"--l {self.data_dirpath} qdess --fc t2 --suppress_fat "
            f"--suppress_fluid --beta 1.1"
        )
        self.__cmd_line_helper__(cmdline_str)


if __name__ == '__main__':
    unittest.main()
