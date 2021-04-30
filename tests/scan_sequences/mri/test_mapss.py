import os
import unittest

from dosma.core.io import NiftiReader
from dosma.scan_sequences import Mapss
from dosma.tissues.femoral_cartilage import FemoralCartilage

from ... import util

SEGMENTATION_WEIGHTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "../../weights/iwoai-2019-t6-normalized"
)
SEGMENTATION_MODEL = "iwoai-2019-t6-normalized"

# Path to manual segmentation mask
MANUAL_SEGMENTATION_MASK_PATH = os.path.join(
    util.get_scan_dirpath(Mapss.NAME), "misc/fc_manual.nii.gz"
)


@unittest.skipIf(not util.is_data_available(), "unittest data is not available")
class MapssTest(util.ScanTest):
    SCAN_TYPE = Mapss

    def test_quant_val_fitting(self):
        """Test quantitative fitting (T1-rho, T2)"""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())

        actions = [scan.generate_t1_rho_map, scan.generate_t2_map]
        for action in actions:
            tissue = FemoralCartilage()
            map1 = action(tissue, MANUAL_SEGMENTATION_MASK_PATH)
            assert map1 is not None, "%s: map1 should not be None" % str(action)

            nr = NiftiReader()
            tissue.set_mask(nr.load(MANUAL_SEGMENTATION_MASK_PATH))
            map2 = action(tissue, num_workers=util.num_workers())
            assert map2 is not None, "%s: map2 should not be None" % str(action)

            # map1 and map2 should be identical
            assert map1.volumetric_map.is_identical(
                map2.volumetric_map
            ), "%s: map1 and map2 should be identical" % str(action)

    @unittest.skipIf(not util.is_elastix_available(), "elastix is not available")
    def test_cmd_line(self):
        # Intraregister
        cmdline_str = "--d %s --s %s mapss intraregister" % (self.dicom_dirpath, self.data_dirpath)
        self.__cmd_line_helper__(cmdline_str)

        # Estimate T1-rho for femoral cartilage.
        cmdline_str = "--l %s mapss --fc t1_rho --mask %s" % (
            self.data_dirpath,
            MANUAL_SEGMENTATION_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2 map for femoral cartilage, tibial cartilage, and meniscus via command line
        cmdline_str = "--l %s mapss --fc t2 --mask %s" % (
            self.data_dirpath,
            MANUAL_SEGMENTATION_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)
