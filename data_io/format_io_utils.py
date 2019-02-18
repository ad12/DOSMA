import os

from data_io.dicom_io import DicomWriter, DicomReader, contains_dicom_extension
from data_io.format_io import SUPPORTED_FORMATS
from data_io.nifti_io import NiftiWriter, NiftiReader, contains_nifti_extension
from data_io.format_io import ImageDataFormat, DataReader, DataWriter


def get_writer(data_format: ImageDataFormat) -> DataWriter:
    if data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == ImageDataFormat.nifti:
        return NiftiWriter()
    elif data_format == ImageDataFormat.dicom:
        return DicomWriter()


def get_reader(data_format: ImageDataFormat) -> DataReader:
    if data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == ImageDataFormat.nifti:
        return NiftiReader()
    elif data_format == ImageDataFormat.dicom:
        return DicomReader()


def get_data_format(file_or_dirname: str) -> ImageDataFormat:
    # if a directory, assume that format is dicom
    filename_base, ext = os.path.splitext(file_or_dirname)
    if ext == '' or contains_dicom_extension(file_or_dirname):
        return ImageDataFormat.dicom

    if contains_nifti_extension(file_or_dirname):
        return ImageDataFormat.nifti

    raise ValueError('This data format supported: %s' % file_or_dirname)


def convert_format_filename(file_or_dirname: str, new_data_format: ImageDataFormat) -> str:
    if new_data_format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    current_format = get_data_format(file_or_dirname)

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


def get_filepath_variations(file_or_dirname: str):
    filepath_variations = []
    for io_format in SUPPORTED_FORMATS:
        filepath_variations.append(convert_format_filename(file_or_dirname, io_format))
    return filepath_variations


def generic_load(file_or_dirname: str, expected_num_volumes=None):
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

    io_format = get_data_format(exist_path)
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
