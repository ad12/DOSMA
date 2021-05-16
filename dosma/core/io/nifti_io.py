"""NIfTI I/O.

This module contains NIfTI input/output helpers.


"""

import os
from typing import Collection

import nibabel as nib

from dosma.core.io.format_io import DataReader, DataWriter, ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION
from dosma.utils import io_utils

__all__ = ["NiftiReader", "NiftiWriter"]


class NiftiReader(DataReader):
    """A class for reading NIfTI files.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.nifti

    def load(self, file_path) -> MedicalVolume:
        """Load volume from NIfTI file path.

        A NIfTI file should only correspond to one volume.

        Args:
            file_path (str): File path to NIfTI file.

        Returns:
            MedicalVolume: Loaded volume.

        Raises:
            FileNotFoundError: If `file_path` not found.
            ValueError: If `file_path` does not end in a supported NIfTI extension.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError("{} not found".format(file_path))

        if not self.data_format_code.is_filetype(file_path):
            raise ValueError(
                "{} must be a file with extension '.nii' or '.nii.gz'".format(file_path)
            )

        nib_img = nib.load(file_path)
        return MedicalVolume.from_nib(
            nib_img,
            affine_precision=AFFINE_DECIMAL_PRECISION,
            origin_precision=SCANNER_ORIGIN_DECIMAL_PRECISION,
        )

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()


class NiftiWriter(DataWriter):
    """A class for writing volumes in NIfTI format.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.nifti

    def save(self, volume: MedicalVolume, file_path: str):
        """Save volume in NIfTI format,

        Args:
            volume (MedicalVolume): Volume to save.
            file_path (str): File path to NIfTI file.

        Raises:
            ValueError: If `file_path` does not end in a supported NIfTI extension.
        """
        if not self.data_format_code.is_filetype(file_path):
            raise ValueError(
                "{} must be a file with extension '.nii' or '.nii.gz'".format(file_path)
            )

        # Create dir if does not exist
        io_utils.mkdirs(os.path.dirname(file_path))

        nib_img = volume.to_nib()
        nib.save(nib_img, file_path)

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()
