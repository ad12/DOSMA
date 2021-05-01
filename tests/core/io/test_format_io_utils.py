import os
import unittest

import nibabel.testing as nib_testing
import pydicom.data as pydd

import dosma.core.io.format_io_utils as fio_utils
from dosma.core.io.dicom_io import DicomReader, DicomWriter
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter


class TestFormatIOUtils(unittest.TestCase):
    def test_get_reader_writer(self):
        assert isinstance(fio_utils.get_reader(ImageDataFormat.nifti), NiftiReader)
        assert isinstance(fio_utils.get_reader(ImageDataFormat.dicom), DicomReader)

        assert isinstance(fio_utils.get_writer(ImageDataFormat.nifti), NiftiWriter)
        assert isinstance(fio_utils.get_writer(ImageDataFormat.dicom), DicomWriter)

    def test_convert_image_data_format(self):
        dcm_dir = "/path/to/dcm/data"
        dcm_fname = "data.dcm"
        nifti_fname = "data.nii.gz"

        assert fio_utils.convert_image_data_format(dcm_dir, ImageDataFormat.dicom) == dcm_dir
        assert (
            fio_utils.convert_image_data_format(dcm_dir, ImageDataFormat.nifti)
            == f"{dcm_dir}.nii.gz"
        )

        assert fio_utils.convert_image_data_format(dcm_fname, ImageDataFormat.dicom) == dcm_fname
        # TODO: Activate when support for single dicom files is added
        # assert fio_utils.convert_image_data_format(dcm_fname, ImageDataFormat.nifti) == nifti_fname  # noqa: E501

        assert (
            fio_utils.convert_image_data_format(nifti_fname, ImageDataFormat.nifti) == nifti_fname
        )
        assert fio_utils.convert_image_data_format(nifti_fname, ImageDataFormat.dicom) == "data"

    def test_get_filepath_variations(self):
        dcm_dir = "/path/to/dcm/data"
        nifti_fname = "data.nii.gz"

        fp_variations = fio_utils.get_filepath_variations(dcm_dir)
        assert set(fp_variations) == {dcm_dir, f"{dcm_dir}.nii.gz"}

        fp_variations = fio_utils.get_filepath_variations(nifti_fname)
        assert set(fp_variations) == {nifti_fname, "data"}

    def test_generic_load(self):
        nib_data = os.path.join(nib_testing.data_path, "example4d.nii.gz")
        vol = fio_utils.generic_load(nib_data)
        expected = NiftiReader().load(nib_data)
        assert vol.is_identical(expected)

        dcm_data = os.path.join(pydd.get_testdata_file("MR_small.dcm"))
        vol = fio_utils.generic_load(dcm_data)[0]
        expected = DicomReader().load(dcm_data)[0]
        assert vol.is_identical(expected)
