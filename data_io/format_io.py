"""
File detailing abstract classes for reading/writing data of different formats

@author: Arjun Desai
        (C) Stanford University, 2019
"""
import enum
import os
from abc import ABC, abstractmethod

__all__ = ['ImageDataFormat', 'DataReader', 'DataWriter', 'SUPPORTED_VISUALIZATION_FORMATS']

SUPPORTED_VISUALIZATION_FORMATS = ('png', 'eps', 'pdf', 'jpeg', 'pgf', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tiff')


class ImageDataFormat(enum.Enum):
    """
    Enum describing supported image data formats for I/O
    """
    nifti = 1, ('nii', 'nii.gz')
    dicom = 2, ('dcm',)

    def __new__(cls, keycode, extensions):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.extensions = extensions
        return obj

    def is_filetype(self, filepath: str) -> bool:
        """
        Check if filepath matches the filetype specified by ImageDataFormat
        This method checks to make sure the extensions are appropriate

        :param filepath: A string
        :return: a boolean
        """
        bool_list = [filepath.endswith('.%s' % ext) for ext in self.extensions]
        return bool(sum(bool_list))

    @classmethod
    def get_data_format(cls, file_or_dirname: str):
        """
        Get the image data format the corresponds best to the filepath
        :param file_or_dirname: a string defining a path to a file or to a directory

        :raises ValueError if no compatible image data format found
        :return: an ImageDataFormat
        """
        for im_data_format in cls:
            if im_data_format.is_filetype(file_or_dirname):
                return im_data_format

        # if no extension found, assume the name corresponds to a directory and assume that format is dicom
        filename_base, ext = os.path.splitext(file_or_dirname)
        if filename_base == file_or_dirname:
            return ImageDataFormat.dicom

        raise ValueError('Unknown data format for %s' % file_or_dirname)


class DataReader(ABC):
    """
    An abstract class for reading medical data
    Format-specific readers should inherit from this class (i.e. DicomReader, NiftiReader)
    """
    data_format_code = None  # must be defined in subclasses (should be type ImageDataFormat)

    @abstractmethod
    def load(self, filepath: str):
        pass


class DataWriter(ABC):
    """
    An abstract class for writing medical data
    Format-specific writers should inherit from this class (i.e. DicomWriter, NiftiWriter)
    """
    data_format_code = None  # must be defined in subclasses (should be type ImageDataFormat)

    from data_io.med_volume import MedicalVolume
    @abstractmethod
    def save(self, im: MedicalVolume, filepath: str):
        pass
