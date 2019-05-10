"""
Utils for data I/O

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import os

from data_io.dicom_io import DicomWriter, DicomReader
from data_io.format_io import DataWriter, DataReader, ImageDataFormat
from data_io.nifti_io import NiftiWriter, NiftiReader

__all__ = ['get_reader', 'get_writer', 'get_filepath_variations', 'convert_image_data_format',
           'generic_load']

__READERS = {ImageDataFormat.dicom: DicomReader, ImageDataFormat.nifti: NiftiReader}
__WRITERS = {ImageDataFormat.dicom: DicomWriter, ImageDataFormat.nifti: NiftiWriter}


def get_reader(data_format: ImageDataFormat) -> DataReader:
    """
    Return a DataReader corresponding to the given data_format

    :param data_format: an Image DataFormat
    :return: a DataReader
    """
    return __READERS[data_format]()


def get_writer(data_format: ImageDataFormat) -> DataWriter:
    """
    Return a DataWriter corresponding to given data_format

    :param data_format: an ImageDataFormat
    :return: a DataWriter
    """
    return __WRITERS[data_format]()


def convert_image_data_format(file_or_dirname: str, new_data_format: ImageDataFormat) -> str:
    """
    Convert a file or directory name given an image data format

    :param file_or_dirname: a filepath or directory path
    :param new_data_format: an ImageDataFormat used to convert file/directory name

    :raises NotImplementedError if conversion from current image data format to new image data format not found

    :return: a string defining file/directory path based on new_data_format
    """
    current_format = ImageDataFormat.get_data_format(file_or_dirname)

    if current_format == new_data_format:
        return file_or_dirname

    if current_format == ImageDataFormat.dicom and new_data_format == ImageDataFormat.nifti:
        dirname = os.path.dirname(file_or_dirname)
        basename = os.path.basename(file_or_dirname)
        return os.path.join(dirname, '%s.nii.gz' % basename)

    if current_format == ImageDataFormat.nifti and new_data_format == ImageDataFormat.dicom:
        dirname = os.path.dirname(file_or_dirname)
        basename = os.path.basename(file_or_dirname)
        basename = basename.split('.', 1)[0]
        return os.path.join(dirname, basename)

    raise NotImplementedError('%s --> %s not implemented.' % (current_format.name, new_data_format.name))


def get_filepath_variations(file_or_dirname: str):
    """
    Get variations in filepath given different image data formats

    :param file_or_dirname: a filepath or directory path
    :return: a list of filepaths corresponding to naming conventions of different ImageDataFormats
    """
    filepath_variations = []
    for io_format in ImageDataFormat:
        filepath_variations.append(convert_image_data_format(file_or_dirname, io_format))
    return filepath_variations


def generic_load(file_or_dirname: str, expected_num_volumes=None):
    """
    Load MedicalVolume(s) from a filepath or directory path regardless of ImageDataFormat

    :param file_or_dirname: a string defining filepath or directory path
    :param expected_num_volumes (optional): an int defining the number of volumes expected. default: None.
                                            if specified, assert if number of loaded volumes != expected num volumes

    :raises ValueError if multiple filepaths corresponding to different ImageDataFormats exist
    :raises FileNotFoundError if filepath or corresponding versions of filepath not found

    :return: a list of MedicalVolumes or a single MedicalVolume if expected_num_volumes == 1
    """
    possible_filepaths = get_filepath_variations(file_or_dirname)
    exist_path = None

    for fp in possible_filepaths:
        if os.path.exists(fp):
            if exist_path is not None:
                raise ValueError(
                    'Ambiguous loading state - multiple possible files to load from %s' % str(possible_filepaths))
            exist_path = fp

    if exist_path is None:
        raise FileNotFoundError('No file associated with basename %s found' % os.path.basename(file_or_dirname))

    io_format = ImageDataFormat.get_data_format(exist_path)
    r = get_reader(io_format)
    vols = r.load(exist_path)

    if expected_num_volumes is None:
        return vols

    if type(vols) is not list:
        vols = [vols]

    assert len(vols) == expected_num_volumes, "Expected %d volumes, got %d" % (expected_num_volumes, len(vols))

    if len(vols) == 1:
        return vols[0]

    return vols
