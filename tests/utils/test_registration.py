import os
import shutil
import unittest

import dosma.file_constants as fc
from dosma.data_io.dicom_io import DicomReader
from dosma.data_io.format_io import ImageDataFormat
from dosma.scan_sequences import CubeQuant, QDess
from dosma.utils.registration import apply_warp, register

from .. import util

QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(CubeQuant.NAME), 'misc/fc.nii.gz')


class TestRegister(unittest.TestCase):
    def test_run(self):
        dr = DicomReader(num_workers=util.num_workers())
        cq_dicoms = util.get_dicoms_path(os.path.join(util.UNITTEST_SCANDATA_PATH, CubeQuant.NAME))
        cq = dr.load(cq_dicoms)
        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-register")
        out, vols = register(
            cq[0], cq[1:], fc.ELASTIX_AFFINE_PARAMS_FILE, out_path, 
            num_workers=util.num_workers(), num_threads=2, return_volumes=True
        )
        shutil.rmtree(out_path)


class TestApplyWarp(unittest.TestCase):
    def test_run(self):
        dr = DicomReader(num_workers=util.num_workers())
        cq_dicoms = util.get_dicoms_path(os.path.join(util.UNITTEST_SCANDATA_PATH, CubeQuant.NAME))
        cq = dr.load(cq_dicoms)
        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-register")
        out, vols = register(
            cq[0], cq[1:], fc.ELASTIX_AFFINE_PARAMS_FILE, out_path, 
            num_workers=util.num_workers(), num_threads=2, return_volumes=True
        )
        apply_warp(cq[0], out_registration=out[0])
        shutil.rmtree(out_path)


if __name__ == "__main__":
    unittest.main()


