from abc import ABC, abstractmethod

import enum

class ImageDataFormat(enum.Enum):
    unsupported = 0
    nifti = 1
    dicom = 2

SUPPORTED_FORMATS = (ImageDataFormat.nifti, ImageDataFormat.dicom)


class DataReader(ABC):
    """
    This is the class for reading in medical data in various formats
    """
    data_format_code = ImageDataFormat.unsupported

    @abstractmethod
    def load(self, filepath: str):
        pass

class DataWriter(ABC):
    """
    This is the class for writing medical data in various formats
    """
    data_format_code = ImageDataFormat.unsupported

    from data_io.med_volume import MedicalVolume
    @abstractmethod
    def save(self, im: MedicalVolume, filepath: str):
        pass
