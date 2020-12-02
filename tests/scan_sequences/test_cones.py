import os
import shutil
import unittest

import numpy as np

import dosma.file_constants as fc
from dosma.data_io import DicomReader, ImageDataFormat, NiftiReader
from dosma.scan_sequences import Cones, QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage
from dosma.utils.registration import apply_warp, register

from .. import util

# target mask path used to register Cubequant volume to qDESS volume
QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(Cones.NAME), 'misc/fc.nii.gz')


class ConesTest(util.ScanTest):
    SCAN_TYPE = Cones

    def test_interregister(self):
        """Test Cones interregistration."""
        # Register to first echo of QDess without a mask.
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH)

        # Register to first echo of QDess with a mask.
        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)
    
    def test_interregister_upgrade_no_mask(self):
        """Verify cones interregistering using new registration.
        
        To be deleted once Cones registration is upgraded
        (https://github.com/ad12/DOSMA/issues/55).
        """
        nr = NiftiReader()

        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH)
        subvolumes = list(scan.subvolumes.values())

        # Inter-register with last echo.
        vols = DicomReader(num_workers=util.num_workers()).load(self.dicom_dirpath)
        base, moving = vols[-1], vols[:-1]

        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-interregister-no-mask")
        out_reg, _ = register(
            QDESS_ECHO1_PATH, base, 
            parameters=[fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE],
            output_path=out_path,
            sequential=True,
            collate=True,
            num_workers=util.num_workers(),
            num_threads=2, 
            return_volumes=False,
            rtype=tuple,
        )
        out_reg = out_reg[0]

        reg_vols = [nr.load(out_reg.warped_file)]
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))
        
        for idx, vol in enumerate(reg_vols):
            assert np.allclose(vol.volume, subvolumes[idx].volume), idx
        
        shutil.rmtree(out_path)
    
    def test_interregister_upgrade_mask(self):
        """Verify cones interregistering using new registration.
        
        To be deleted once Cones registration is upgraded
        (https://github.com/ad12/DOSMA/issues/55).
        """
        nr = NiftiReader()

        scan = self.SCAN_TYPE(dicom_path=self.dicom_dirpath)
        scan.interregister(target_path=QDESS_ECHO1_PATH, target_mask_path=TARGET_MASK_PATH)
        subvolumes = list(scan.subvolumes.values())

        # Inter-register with last echo.
        vols = DicomReader(num_workers=util.num_workers()).load(self.dicom_dirpath)
        base, moving = vols[-1], vols[:-1]
        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-interregister-mask")

        # File has to be dilated.
        mask_path = scan.__dilate_mask__(TARGET_MASK_PATH, out_path)

        out_reg, _ = register(
            QDESS_ECHO1_PATH, base, 
            parameters=[fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE],
            output_path=out_path,
            sequential=True,
            collate=True,
            num_workers=util.num_workers(),
            num_threads=2, 
            return_volumes=False,
            target_mask=mask_path,
            use_mask=[False, True],
            rtype=tuple,
        )
        out_reg = out_reg[0]

        reg_vols = [nr.load(out_reg.warped_file)]
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))
        
        for idx, vol in enumerate(reg_vols):
            assert np.allclose(vol.volume, subvolumes[idx].volume), idx
        
        shutil.rmtree(out_path)

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
