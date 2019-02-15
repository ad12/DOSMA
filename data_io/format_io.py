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