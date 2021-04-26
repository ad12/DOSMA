import os
import unittest

from dosma.core.io import ImageDataFormat, NiftiReader
from dosma.scan_sequences import CubeQuant, QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage

from .. import util

# target mask path used to register Cubequant volume to qDESS volume
QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(CubeQuant.NAME), "misc/fc.nii.gz")


class CubeQuantTest(util.ScanTest):
    SCAN_TYPE = CubeQuant

    def test_interregister_no_mask(self):
        """Register Cubequant scan to qDESS scan without a target mask"""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())

        # Register to first echo of QDess without a mask
        scan.interregister(target_path=QDESS_ECHO1_PATH)

    def test_interregister_mask(self):
        """Register Cubequant scan to qDESS scan with a target mask (mask for femoral cartilage)"""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

    def test_t1_rho_map(self):
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

        # run analysis with femoral cartilage, without mask
        tissue = FemoralCartilage()
        map1 = scan.generate_t1_rho_map(tissue, TARGET_MASK_PATH, num_workers=util.num_workers())
        assert map1 is not None, "map should not be None"

        # add mask to femoral cartilage and run
        nr = NiftiReader()
        tissue.set_mask(nr.load(TARGET_MASK_PATH))
        map2 = scan.generate_t1_rho_map(tissue, num_workers=util.num_workers())
        assert map2 is not None, "map should not be None"

        # map1 and map2 should be identical
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

    def test_intraregister(self):
        """Verify cubequant intraregistering using new registration."""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        scan.intraregister()

    def test_cmd_line(self):
        # Intraregister
        cmdline_str = "--d %s --s %s cubequant intraregister" % (
            self.dicom_dirpath,
            self.data_dirpath,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate segmentation mask for femoral cartilage via command line
        cmdline_str = "--l %s cubequant --fc interregister --tp %s --tm %s" % (
            self.data_dirpath,
            QDESS_ECHO1_PATH,
            TARGET_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T1rho map for femoral cartilage, tibial cartilage, and meniscus via command line
        cmdline_str = "--l %s cubequant --fc t1_rho --mask_path %s" % (
            self.data_dirpath,
            TARGET_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)


if __name__ == "__main__":
    unittest.main()
