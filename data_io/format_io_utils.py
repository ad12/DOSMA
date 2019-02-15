from data_io.dicom_io import DicomWriter, DicomReader, __DICOM_EXTENSIONS__, contains_dicom_extension
from data_io.format_io import SUPPORTED_FORMATS
from data_io.nifti_io import NiftiWriter, NiftiReader, __NIFTI_EXTENSIONS__, contains_nifti_extension

import os


def get_writer(data_format: str):
    if data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == 'nifti':
        return NiftiWriter()
    elif data_format == 'dicom':
        return DicomWriter()

def get_reader(data_format: str):
    if data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == 'nifti':
        return NiftiReader()
    elif data_format == 'dicom':
        return DicomReader()

def get_data_format(file_or_dirname: str):
    # if a directory, assume that format is dicom
    filename_base, ext = os.path.splitext(file_or_dirname)
    if ext == '' or contains_dicom_extension(file_or_dirname):
        return 'dicom'

    if contains_nifti_extension(file_or_dirname):
        return 'nifti'

    raise ValueError('This data format supported: %s' % file_or_dirname)


def convert_format_filename(file_or_dirname: str, new_data_format: str):
    if new_data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    current_format = get_data_format(file_or_dirname)

    if current_format == new_data_format:
        return file_or_dirname

    if current_format=='dicom' and new_data_format == 'nifti':
        dirname = os.path.dirname(file_or_dirname)
        basename = os.path.basename(file_or_dirname)
        return os.path.join(dirname, '%s.nii.gz' % basename)

    if current_format=='nifti' and new_data_format == 'dicom':
        dirname = os.path.dirname(file_or_dirname)
        basename = os.path.basename(file_or_dirname)
        basename, ext = os.path.splitext(basename)
        return os.path.join(dirname, basename)


def get_filepath_variations(file_or_dirname: str):
    filepath_variations = []
    for io_format in SUPPORTED_FORMATS:
        filepath_variations.append(convert_format_filename(file_or_dirname, io_format))
    return filepath_variations


def generic_load(file_or_dirname):
    possible_filepaths = get_filepath_variations(file_or_dirname)
    exist_path = None

    for fp in possible_filepaths:
        if os.path.exists(fp):
            if exist_path is not None:
                raise ValueError('Ambiguous loading state - multiple possible files to load from %s' % str(possible_filepaths))
            exist_path = fp

    if exist_path is None:
        raise FileNotFoundError('No file associated with basename %s found' % os.path.basename(file_or_dirname))

    io_format = get_data_format(exist_path)
    r = get_reader(io_format)
    return r.load(exist_path)
