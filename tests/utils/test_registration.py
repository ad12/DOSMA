import multiprocessing as mp
import os
import shutil
import unittest
from functools import partial

import numpy as np

import dosma.file_constants as fc
from dosma.data_io.dicom_io import DicomReader
from dosma.data_io.format_io import ImageDataFormat
from dosma.scan_sequences import CubeQuant, QDess
from dosma.utils.registration import apply_warp, register

from .. import util

QDESS_ECHO1_PATH = util.get_read_paths(util.get_scan_dirpath(QDess.NAME), ImageDataFormat.nifti)[0]
TARGET_MASK_PATH = os.path.join(util.get_scan_dirpath(CubeQuant.NAME), 'misc/fc.nii.gz')


class TestRegister(unittest.TestCase):
    def test_multiprocessing(self):
        dr = DicomReader(num_workers=util.num_workers())
        cq_dicoms = util.get_dicoms_path(os.path.join(util.UNITTEST_SCANDATA_PATH, CubeQuant.NAME))
        cq = dr.load(cq_dicoms)
        data_dir = os.path.join(fc.TEMP_FOLDER_PATH, "test-register-mp")

        out_path = os.path.join(data_dir, "expected")
        _, expected = register(
            cq[0], cq[1:], fc.ELASTIX_AFFINE_PARAMS_FILE, out_path, 
            num_workers=0, num_threads=2, return_volumes=True, rtype=tuple,
            show_pbar=True
        )

        out_path = os.path.join(data_dir, "out")
        _, out = register(
            cq[0], cq[1:], fc.ELASTIX_AFFINE_PARAMS_FILE, out_path, 
            num_workers=util.num_workers(), num_threads=2, return_volumes=True, rtype=tuple,
            show_pbar=True
        )

        for vol, exp in zip(out, expected):
            assert np.allclose(vol.volume, exp.volume)

        shutil.rmtree(data_dir)


class TestApplyWarp(unittest.TestCase):    
    def test_multiprocessing(self):
        """Verify that multiprocessing compatible with apply_warp."""
        # Generate viable transform file.
        dr = DicomReader(num_workers=util.num_workers())
        cq_dicoms = util.get_dicoms_path(os.path.join(util.UNITTEST_SCANDATA_PATH, CubeQuant.NAME))
        cq = dr.load(cq_dicoms)
        out_path = os.path.join(fc.TEMP_FOLDER_PATH, "test-register")
        out, _ = register(
            cq[0], cq[1], fc.ELASTIX_AFFINE_PARAMS_FILE, out_path, 
            num_workers=util.num_workers(), num_threads=2, return_volumes=False, rtype=tuple,
        )
        vols = cq[2:]

        # Single process (main thread)
        expected = []
        for v in vols:
            expected.append(apply_warp(v, out_registration=out[0]))

        # Multiple process
        func = partial(apply_warp, out_registration=out[0])
        with mp.Pool(min(len(vols), util.num_workers())) as p:
            outputs = p.map(func, vols)
        
        for out, exp in zip(outputs, expected):
            assert np.allclose(out.volume, exp.volume)

        shutil.rmtree(out_path)


if __name__ == "__main__":
    unittest.main()


