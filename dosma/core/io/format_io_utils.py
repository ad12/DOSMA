"""Utils for data I/O.
"""

import os
from pathlib import Path
from typing import Union

from dosma.core.io.dicom_io import DicomReader, DicomWriter
from dosma.core.io.format_io import DataReader, DataWriter, ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter

__all__ = [
    "get_reader",
    "get_writer",
    "get_filepath_variations",
    "convert_image_data_format",
    "generic_load",
]

_READERS = {ImageDataFormat.dicom: DicomReader, ImageDataFormat.nifti: NiftiReader}
_WRITERS = {ImageDataFormat.dicom: DicomWriter, ImageDataFormat.nifti: NiftiWriter}


def get_reader(data_format: ImageDataFormat) -> DataReader:
    """Return a DataReader corresponding to the given data format.

    Args:
        data_format (ImageDataFormat): Data format to read.

    Returns:
        DataReader: Reader for given format.
    """
    return _READERS[data_format]()


def get_writer(data_format: ImageDataFormat) -> DataWriter:
    """Return a DataWriter corresponding to given data format.

    Args:
        data_format (ImageDataFormat): Data format to write.

    Returns:
        DataWriter: Writer for given format.
    """
    return _WRITERS[data_format]()


def convert_image_data_format(
    file_or_dir_path: Union[str, Path, os.PathLike], new_data_format: ImageDataFormat
) -> str:
    """Change a file or directory name to convention of another data format.

    Args:
        file_or_dir_path (str): File or directory path.
        new_data_format (ImageDataFormat): Data format convention for file/directory name.

    Returns:
        str: File/directory path based on convention of new data format.

    Raises:
        NotImplementedError: If conversion from current image data format
            to new image data format not found.
    """
    current_format = ImageDataFormat.get_image_data_format(file_or_dir_path)

    if current_format == new_data_format:
        return file_or_dir_path

    if current_format == ImageDataFormat.dicom and new_data_format == ImageDataFormat.nifti:
        dirname = os.path.dirname(file_or_dir_path)
        basename = os.path.basename(file_or_dir_path)
        return os.path.join(dirname, "{}.nii.gz".format(basename))

    if current_format == ImageDataFormat.nifti and new_data_format == ImageDataFormat.dicom:
        dirname = os.path.dirname(file_or_dir_path)
        basename = os.path.basename(file_or_dir_path)
        basename = basename.split(".", 1)[0]
        return os.path.join(dirname, basename)

    raise NotImplementedError(
        "{} -> {} not implemented".format(current_format.name, new_data_format.name)
    )


def get_filepath_variations(file_or_dir_path: Union[str, Path, os.PathLike]):
    """Get file paths using convention for all different image data formats.

    Args:
        file_or_dir_path (str): File or directory path.

    Returns:
        list[str]: File paths corresponding to naming conventions of different ImageDataFormats.
    """
    filepath_variations = []
    for io_format in ImageDataFormat:
        filepath_variations.append(convert_image_data_format(file_or_dir_path, io_format))
    return filepath_variations


def generic_load(file_or_dir_path: Union[str, Path, os.PathLike], expected_num_volumes: int = None):
    """Load MedicalVolume(s) from a file or directory without knowing data format.

    Args:
        file_or_dir_path (str): File or directory path.
        expected_num_volumes (int, `optional`): Number of volumes expected.
            If specified, assert if number of loaded
            volumes != expected num volumes. Defaults to `None`.

    Returns:
        `list[MedicalVolume]` or `MedicalVolume`: Volume(s) loaded.
            If ``expected_num_volumes = 1``, returns :class:``MedicalVolume``.

    Raises:
        ValueError: If multiple file paths corresponding to different ImageDataFormats exist.
        FileNotFoundError: If file path or corresponding versions of file path not found.
    """
    possible_filepaths = get_filepath_variations(file_or_dir_path)
    exist_path = None

    for fp in possible_filepaths:
        if os.path.exists(fp):
            if exist_path is not None:
                raise ValueError(
                    "Ambiguous loading state - multiple possible files to load from %s"
                    % str(possible_filepaths)
                )
            exist_path = fp

    if exist_path is None:
        raise FileNotFoundError(
            "No file associated with basename %s found" % os.path.basename(file_or_dir_path)
        )

    io_format = ImageDataFormat.get_image_data_format(exist_path)
    r = get_reader(io_format)
    vols = r.load(exist_path)

    if expected_num_volumes is None:
        return vols

    if type(vols) is not list:
        vols = [vols]

    assert len(vols) == expected_num_volumes, "Expected %d volumes, got %d" % (
        expected_num_volumes,
        len(vols),
    )

    if len(vols) == 1:
        return vols[0]

    return vols
