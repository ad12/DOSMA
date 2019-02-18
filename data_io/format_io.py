import enum
from abc import ABC, abstractmethod


class ImageDataFormat(enum.Enum):
    """
    Enum describing image data formats for IO
    """
    unsupported = 0
    nifti = 1
    dicom = 2


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
