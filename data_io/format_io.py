"""
File detailing abstract classes for reading/writing data of different formats

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import enum
from abc import ABC, abstractmethod


__all__ = ['ImageDataFormat', 'SUPPORTED_FORMATS', 'DataReader', 'DataWriter']


class ImageDataFormat(enum.Enum):
    """
    Enum describing image data formats for IO
    """
    unsupported = 0
    nifti = 1
    dicom = 2


# These formats are currently supported for reading/writing volumes
SUPPORTED_FORMATS = (ImageDataFormat.nifti, ImageDataFormat.dicom)


class DataReader(ABC):
    """
    An abstract class for reading medical data
    Format-specific readers should inherit from this class (i.e. DicomReader, NiftiReader)
    """
    data_format_code = ImageDataFormat.unsupported

    @abstractmethod
    def load(self, filepath: str):
        pass


class DataWriter(ABC):
    """
    An abstract class for writing medical data
    Format-specific writers should inherit from this class (i.e. DicomWriter, NiftiWriter)
    """
    data_format_code = ImageDataFormat.unsupported

    from data_io.med_volume import MedicalVolume
    @abstractmethod
    def save(self, im: MedicalVolume, filepath: str):
        pass
