import unittest

from dosma.core.io.format_io import ImageDataFormat


class TestImageDataFormat(unittest.TestCase):
    def test_isfiletype(self):
        dcm_fname = "data.dcm"
        nifti_fname = "data.nii.gz"

        assert ImageDataFormat.dicom.is_filetype(dcm_fname)
        assert not ImageDataFormat.dicom.is_filetype(nifti_fname)

        assert ImageDataFormat.nifti.is_filetype(nifti_fname)
        assert not ImageDataFormat.nifti.is_filetype(dcm_fname)

    def get_image_data_format(self):
        dcm_fname = "data.dcm"
        dcm_dir = "/path/to/dir"
        nifti_fname = "data.nii.gz"

        assert ImageDataFormat.get_image_data_format(dcm_fname) == ImageDataFormat.dicom
        assert ImageDataFormat.get_image_data_format(dcm_dir) == ImageDataFormat.dicom
        assert ImageDataFormat.get_image_data_format(nifti_fname) == nifti_fname
