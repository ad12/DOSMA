import multiprocessing as mp
import os
import shutil
import unittest
from functools import partial

import numpy as np

import dosma.file_constants as fc
from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import to_affine
from dosma.core.registration import apply_warp, register

from .. import util


def _generate_translated_vols(n=3):
    """Generate mock data that is translated diagonally by 1 pixel."""
    mvs = []
    affine = to_affine(("SI", "AP"), (0.3, 0.3, 0.5))
    for offset in range(n):
        arr = np.zeros((250, 250, 10))
        arr[15 + offset : 35 + offset, 15 + offset : 35 + offset] = 1
        mvs.append(MedicalVolume(arr, affine))
    return mvs


class TestRegister(util.TempPathMixin):
    @unittest.skipIf(not util.is_elastix_available(), "elastix is not available")
    def test_multiprocessing(self):
        mvs = _generate_translated_vols()
        data_dir = os.path.join(self.data_dirpath, "test-register-mp")

        out_path = os.path.join(data_dir, "expected")
        _, expected = register(
            mvs[0],
            mvs[1:],
            fc.ELASTIX_AFFINE_PARAMS_FILE,
            out_path,
            num_workers=0,
            num_threads=2,
            return_volumes=True,
            rtype=tuple,
            show_pbar=True,
        )

        out_path = os.path.join(data_dir, "out")
        _, out = register(
            mvs[0],
            mvs[1:],
            fc.ELASTIX_AFFINE_PARAMS_FILE,
            out_path,
            num_workers=util.num_workers(),
            num_threads=2,
            return_volumes=True,
            rtype=tuple,
            show_pbar=True,
        )

        for vol, exp in zip(out, expected):
            assert np.allclose(vol.volume, exp.volume)

        shutil.rmtree(data_dir)

    def test_complex(self):
        mvs = _generate_translated_vols()
        mask = mvs[0]._partial_clone(volume=np.ones(mvs[0].shape))
        data_dir = os.path.join(self.data_dirpath, "test-register-complex-sequential-moving-masks")

        out_path = os.path.join(data_dir, "expected")
        _ = register(
            mvs[0],
            mvs[1:],
            [fc.ELASTIX_AFFINE_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE],
            out_path,
            target_mask=mask,
            use_mask=[True, True],
            sequential=True,
            num_workers=0,
            num_threads=2,
            return_volumes=True,
            rtype=tuple,
            show_pbar=True,
        )

        shutil.rmtree(data_dir)


class TestApplyWarp(util.TempPathMixin):
    @unittest.skipIf(not util.is_elastix_available(), "elastix is not available")
    def test_multiprocessing(self):
        """Verify that multiprocessing compatible with apply_warp."""
        # Generate viable transform file.
        mvs = _generate_translated_vols(n=4)
        out_path = os.path.join(self.data_dirpath, "test-apply-warp")
        out, _ = register(
            mvs[0],
            mvs[1],
            fc.ELASTIX_AFFINE_PARAMS_FILE,
            out_path,
            num_workers=util.num_workers(),
            num_threads=2,
            return_volumes=False,
            rtype=tuple,
        )
        vols = mvs[2:]

        # Single process (main thread)
        expected = []
        for v in vols:
            expected.append(apply_warp(v, out_registration=out[0]))

        # Multiple process (within apply warp)
        num_workers = min(len(vols), util.num_workers())
        outputs = apply_warp(vols, out_registration=out[0], num_workers=num_workers)
        for mv_out, exp in zip(outputs, expected):
            assert np.allclose(mv_out.volume, exp.volume)

        # Multiple process
        func = partial(apply_warp, out_registration=out[0])
        with mp.Pool(num_workers) as p:
            outputs = p.map(func, vols)

        for mv_out, exp in zip(outputs, expected):
            assert np.allclose(mv_out.volume, exp.volume)

        shutil.rmtree(out_path)


if __name__ == "__main__":
    unittest.main()
