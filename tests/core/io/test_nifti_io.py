import os
import unittest

import nibabel as nib
import nibabel.testing as nib_testing
import numpy as np

from dosma.core.io.format_io import ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter

from ... import util as ututils


class TestNiftiIO(unittest.TestCase):
    nr = NiftiReader()
    nw = NiftiWriter()

    data_format = ImageDataFormat.nifti

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_nifti_read(self):
        for dp in ututils.SCAN_DIRPATHS:
            dicoms_path = ututils.get_dicoms_path(dp)
            read_filepaths = ututils.get_read_paths(dp, self.data_format)

            for read_filepath in read_filepaths:
                _ = self.nr.load(read_filepath)

                with self.assertRaises(FileNotFoundError):
                    _ = self.nr.load(os.path.join(dp, "bleh"))

                with self.assertRaises(FileNotFoundError):
                    _ = self.nr.load(dp)

                with self.assertRaises(ValueError):
                    _ = self.nr.load(os.path.join(dicoms_path, "I0002.dcm"))

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_nifti_write(self):
        for dp in ututils.SCAN_DIRPATHS:
            read_filepaths = ututils.get_read_paths(dp, self.data_format)
            save_dirpath = ututils.get_write_path(dp, self.data_format)

            for rfp in read_filepaths:
                save_filepath = os.path.join(save_dirpath, os.path.basename(rfp))
                mv = self.nr.load(rfp)
                self.nw.save(mv, save_filepath)

                # cannot save with extensions other than nii or nii.gz
                with self.assertRaises(ValueError):
                    self.nw.save(mv, os.path.join(ututils.TEMP_PATH, "eg.dcm"))

    def test_nifti_nib(self):
        """Test with nibabel sample data."""
        filepath = os.path.join(nib_testing.data_path, "example4d.nii.gz")
        mv_nib = nib.load(filepath)

        nr = NiftiReader()
        mv = nr(filepath)

        assert mv.shape == mv_nib.shape
        assert np.all(mv.A == mv_nib.get_fdata())
        assert np.allclose(mv.affine, mv_nib.affine, atol=1e-4)

        out_path = os.path.join(ututils.TEMP_PATH, "nifti_nib_example.nii.gz")
        nw = NiftiWriter()
        nw(mv, out_path)

        mv_nib2 = nib.load(out_path)
        assert np.all(mv_nib2.get_fdata() == mv_nib.get_fdata())

    def test_state(self):
        nr1 = NiftiReader()
        state_dict = nr1.state_dict()
        state_dict = {k: "foo" for k in state_dict}  # noqa C420

        nr2 = NiftiReader()
        nr2.load_state_dict(state_dict)
        for k in state_dict:
            assert getattr(nr2, k) == "foo"

        nw1 = NiftiWriter()
        state_dict = nw1.state_dict()
        state_dict = {k: "bar" for k in state_dict}  # noqa C420

        nw2 = NiftiWriter()
        nw2.load_state_dict(state_dict)
        for k in state_dict:
            assert getattr(nw2, k) == "bar"

        with self.assertRaises(AttributeError):
            nw2.load_state_dict({"foobar": "delta"})


if __name__ == "__main__":
    unittest.main()
