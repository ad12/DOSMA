import os
import unittest

import file_constants as fc
import pipeline
from tissues.femoral_cartilage import FemoralCartilage
from utils import io_utils


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



if __name__ == '__main__':
    pass
    # unittest.main()
