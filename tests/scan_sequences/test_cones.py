import os
import unittest

from dosma.data_io import ImageDataFormat, NiftiReader
from dosma.scan_sequences import Cones, QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage
from .. import util

# target mask path used to register Cubequant volume to qDESS volume
QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(Cones.NAME), 'misc/fc.nii.gz')


class ConesTest(util.ScanTest):
    SCAN_TYPE = Cones

    def test_loading(self):
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

    def test_interregister(self):
        """Test Cones interregistration."""
        # Register to first echo of QDess without a mask.
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH)

        # Register to first echo of QDess with a mask.
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

    def test_t2_star_map(self):
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)

        # run analysis with femoral cartilage, without mask in tissue, but mask as additional input.
        tissue = FemoralCartilage()
        map1 = scan.generate_t2_star_map(tissue, TARGET_MASK_PATH, num_workers=util.num_workers())
        assert map1 is not None, "map should not be None"

        # add mask to femoral cartilage and run
        nr = NiftiReader()
        tissue.set_mask(nr.load(TARGET_MASK_PATH))
        map2 = scan.generate_t2_star_map(tissue, num_workers=util.num_workers())
        assert map2 is not None, "map should not be None"

        # map1 and map2 should be identical
        assert (map1.volumetric_map.is_identical(map2.volumetric_map))

    def test_cmd_line(self):
        # Generate segmentation mask for femoral cartilage via command line.
        cmdline_str = '--d {} --s {} cones --fc interregister --tp {} --tm {}'.format(self.dicom_dirpath,
                                                                                      self.data_dirpath,
                                                                                      QDESS_ECHO1_PATH,
                                                                                      TARGET_MASK_PATH)
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2star map for femoral cartilage, tibial cartilage, and meniscus via command line.
        cmdline_str = '--l {} cones --fc t2_star --mask_path {}'.format(self.data_dirpath, TARGET_MASK_PATH)
        self.__cmd_line_helper__(cmdline_str)


if __name__ == '__main__':
    unittest.main()
