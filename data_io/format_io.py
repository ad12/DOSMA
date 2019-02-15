from abc import ABC, abstractmethod

SUPPORTED_FORMATS = ('nifti', 'dicom')


class DataReader(ABC):
    """
    This is the class for reading in medical data in various formats
    """

    @abstractmethod
    def load(self, filepath):
        pass


class DataWriter():
    """
    This is the class for writing medical data in various formats
    """

    @abstractmethod
    def load(self, im, filepath):
        pass


from data_io.nifti_io import NiftiReader, NiftiWriter
from data_io.dicom_io import DicomReader, DicomWriter


def get_writer(data_format: str):
    if format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == 'nifti':
        return NiftiWriter()
    elif data_format == 'dicom':
        return DicomWriter()