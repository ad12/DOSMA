"""DICOM I/O.

This module contains DICOM input/output helpers.

Note:

    1. Dicom utilizes LPS convention:
        - LPS: right --> left, anterior --> posterior, inferior --> superior
        - we will call it LPS+, such that letters correspond to increasing end of axis

Attributes:

    TOTAL_NUM_ECHOS_KEY (tuple[int]): Hexadecimal encoding of DICOM tag corresponding to number of echos.
"""

import functools
import multiprocessing as mp
import os
from math import log10, ceil

import nibabel as nib
import numpy as np
import pydicom
from natsort import natsorted

from dosma.data_io import orientation as stdo
from dosma.data_io.format_io import DataReader, DataWriter, ImageDataFormat
from dosma.data_io.med_volume import MedicalVolume
from dosma.defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION
from dosma.utils import io_utils

from typing import List, Union

__all__ = ["DicomReader", "DicomWriter"]

TOTAL_NUM_ECHOS_KEY = (0x19, 0x107e)


def __update_np_dtype__(arr: np.ndarray, bit_depth: int):
    """Create copy of np_array with bit-depth and type specified here.

    Note pydicom only supports writing dicoms with bit-depth 8/16 - only supports bit depths 8/16.

    Args:
        arr (np.ndarray): Numpy array to put into given bit depth.
        bit_depth (int): Bit depth for writing dicom. Must be either `8` or `16`.

    Returns:
        np.ndarray: Copy of input np_array.

    Raises:
        ValueError: If `arr` out of bit-depth range.
        TypeError: If `arr` contains float values out of supported float types.
    """
    assert bit_depth in [8, 16], "Only bit-depths of 8 and 16 are currently supported."
    dtype_dict = {8: [(np.int8, -128, 127), (np.uint8, 0, 255)],
                  16: [(np.float16, -6.55e4, 6.55e4 - 1), (np.int16, -2 ** 15, 2 ** 15), (np.uint16, 0, 2 ** 16 - 1)]}
    supported_floats = [np.float16]
    curr_min = np.min(arr)
    curr_max = np.max(arr)
    contains_float = (arr % 1 != 0).any()

    dtypes = dtype_dict[bit_depth]

    new_dtype = None
    for dtype, dtype_min, dtype_max in dtypes:
        if curr_min < dtype_min or curr_max > dtype_max:
            continue
        new_dtype = dtype
        break
    if not new_dtype:
        raise ValueError("Cannot cast numpy array ({}) to bit-depth of {} bits".format(str(arr.dtype), bit_depth))

    if contains_float and new_dtype not in supported_floats:
        raise TypeError("Array contains float. Cannot cast to numpy array ({}) to {}".format(str(arr.dtype),
                                                                                             new_dtype))

    return arr.astype(new_dtype)


def LPSplus_to_RASplus(headers: List[pydicom.FileDataset]):
    """Convert from LPS+ orientation (default for DICOM) to RAS+ standardized orientation.

    Args:
        headers (list[pydicom.FileDataset]): Headers for DICOM files to reorient. Files should correspond to single
            volume.

    Returns:
        np.ndarray: Affine matrix.
    """
    im_dir = headers[0].ImageOrientationPatient
    in_plane_pixel_spacing = headers[0].PixelSpacing

    orientation = np.zeros([3, 3])

    # Determine vector for in-plane pixel directions (i, j).
    i_vec, j_vec = np.asarray(im_dir[:3]).astype(np.float64), np.asarray(im_dir[3:]).astype(
        np.float64)  # unique to pydicom, please revise if using different library to load dicoms
    i_vec, j_vec = np.round(i_vec, AFFINE_DECIMAL_PRECISION), np.round(j_vec, AFFINE_DECIMAL_PRECISION)
    i_vec = i_vec * in_plane_pixel_spacing[0]
    j_vec = j_vec * in_plane_pixel_spacing[1]

    # Determine vector for through-plane pixel direction (k).
    # 1. Normalize k_vector by magnitude.
    # 2. Multiply by magnitude given by SpacingBetweenSlices field.
    # These actions are done to avoid rounding errors that might result from float subtraction.
    k_vec = np.asarray(headers[1].ImagePositionPatient).astype(np.float64) - np.asarray(
        headers[0].ImagePositionPatient).astype(np.float64)
    k_vec = np.round(k_vec, AFFINE_DECIMAL_PRECISION)

    orientation[:3, :3] = np.stack([j_vec, i_vec, k_vec], axis=1)
    scanner_origin = np.array(headers[0].ImagePositionPatient).astype(np.float64)
    scanner_origin = np.round(scanner_origin, SCANNER_ORIGIN_DECIMAL_PRECISION)

    affine = np.zeros([4, 4])
    affine[:3, :3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2, :] = -1 * affine[:2, :]
    affine[3, 3] = 1

    affine[affine == 0] = 0

    return affine


def _write_dicom_file(np_slice: np.ndarray, header: pydicom.FileDataset, file_path: str):
    """Replace data in header with 2D numpy array and write to `file_path`.

    Args:
        np_slice (np.ndarray): 2D slice to encode in dicom file.
        header (pydicom.FileDataset): DICOM header.
        file_path: File path to write to.
    """
    expected_dimensions = header.Rows, header.Columns
    assert np_slice.shape == expected_dimensions, \
        "In-plane dimension mismatch - expected shape {}, got {}".format(str(expected_dimensions),
                                                                            str(np_slice.shape))

    np_slice_bytes = np_slice.tobytes()
    bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)
    if bit_depth != header.BitsAllocated:
        np_slice = __update_np_dtype__(np_slice, header.BitsAllocated)
        np_slice_bytes = np_slice.tobytes()
        bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)

    assert bit_depth == header.BitsAllocated, \
        "Bit depth mismatch: Expected {:d} got {:d}".format(header.BitsAllocated, bit_depth)

    header.PixelData = np_slice_bytes

    header.save_as(file_path)


class DicomReader(DataReader):
    """A class for reading DICOM files.
    """
    data_format_code = ImageDataFormat.dicom

    def __init__(self, num_workers: int = 0):
        """
        Args:
            num_workers (int, optional): Number of workers to use for loading.
        """
        self.num_workers = num_workers

    def load(self, dicom_dir_path: str, group_by: Union[str, tuple] = 'EchoNumbers', ignore_ext: bool = False):
        """Load dicoms into `MedicalVolume`s grouped by `group_by` tag.

        Args:
            dicom_dir_path (str): Path to directory with dicom files.
            group_by (`str` or `tuple`, optional): DICOM field tag name or tag number used to group dicoms. Defaults
                to `EchoNumbers`. Most DICOM headers encode different echo numbers as volumes acquired at different echo
                times or different phases.
            ignore_ext (`bool`, optional): Ignore extension (`.dcm`) when loading dicoms. Defaults to `False`.

        Returns:
            list[MedicalVolume]: Different volumes grouped by the `group_by` DICOM tag.

        Raises:
            NotADirectoryError: If `dicom_dir_path` does not exist or is not a directory.
            FileNotFoundError: If no dicom files found in directory.

        Note:
            This function sorts files using natsort, an intelligent sorting tool. Please verify dicoms are labeled in a
                sequenced manner (e.g.: dicom1,dicom2,dicom3,...).
        """
        if not os.path.isdir(dicom_dir_path):
            raise NotADirectoryError("Directory {} does not exist".format(dicom_dir_path))

        if not group_by:
            raise ValueError("`group_by` must be specified, even if there are not multiple volumes encoded in dicoms")

        possible_files = os.listdir(dicom_dir_path)

        lstFilesDCM = []
        for f in possible_files:
            # If ignore extension, don't look for '.dcm' extension.
            match_ext = ignore_ext or (not ignore_ext and self.data_format_code.is_filetype(f))
            is_file = os.path.isfile(os.path.join(dicom_dir_path, f))
            is_hidden_file = f.startswith('.')
            if is_file and match_ext and not is_hidden_file:
                lstFilesDCM.append(os.path.join(dicom_dir_path, f))

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No files found in directory {}".format(dicom_dir_path))

        # Check if dicom file has the group_by element specified
        temp_dicom = pydicom.read_file(lstFilesDCM[0], force=True)

        if not temp_dicom.get(group_by):
            raise ValueError("Tag {} does not exist in dicom".format(group_by))

        dicom_data = {}

        if self.num_workers:
            with mp.Pool(self.num_workers) as p:
                dicom_slices = p.map(functools.partial(pydicom.read_file, force=True), lstFilesDCM)
        else:
            dicom_slices = [pydicom.read_file(fp, force=True) for fp in lstFilesDCM]

        for ds in dicom_slices:
            val_groupby = ds.get(group_by)
            if type(val_groupby) is pydicom.DataElement:
                val_groupby = val_groupby.value

            if val_groupby not in dicom_data.keys():
                dicom_data[val_groupby] = {"headers": [], "arr": []}

            dicom_data[val_groupby]["headers"].append(ds)
            dicom_data[val_groupby]["arr"].append(ds.pixel_array)

        vols = []
        for k in sorted(list(dicom_data.keys())):
            dd = dicom_data[k]
            headers = dd["headers"]
            if len(headers) == 0:
                continue
            arr = np.stack(dd["arr"], axis=-1)

            affine = LPSplus_to_RASplus(headers)

            vol = MedicalVolume(arr,
                                affine,
                                headers=headers)
            vols.append(vol)

        return vols


class DicomWriter(DataWriter):
    """A class for writing volumes in DICOM format.
    """
    data_format_code = ImageDataFormat.dicom

    def __init__(self, num_workers: int = 0):
        """
        Args:
            num_workers (int, optional): Number of workers to use for writing.
        """
        self.num_workers = num_workers

    def save(self, volume: MedicalVolume, dir_path: str):
        """Save `medical volume` in dicom format.

        Args:
            volume (MedicalVolume): Volume to save.
            dir_path: Directory path to store dicom files. Dicoms are stored in directories, as multiple files are
                needed to store the volume.

        Raises:
            ValueError: If `im` does not have initialized headers. Or if `im` was flipped across any axis. Flipping
                changes scanner origin, which is currently not handled.
        """
        # Get orientation indicated by headers.
        headers = volume.headers
        if headers is None:
            raise ValueError("MedicalVolume headers must be initialized to save as a dicom")

        affine = LPSplus_to_RASplus(headers)
        orientation = stdo.orientation_nib_to_standard(nib.aff2axcodes(affine))

        # Currently do not support mismatch in scanner_origin.
        if tuple(affine[:3, 3]) != volume.scanner_origin:
            raise ValueError("Scanner origin mismatch. "
                             "Currently we do not handle mismatch in scanner origin (i.e. cannot flip across axis)")

        # Reformat medical volume to expected orientation specified by dicom headers.
        # Store original orientation so we can undo the dicom-specific reformatting.
        original_orientation = volume.orientation

        volume.reformat(orientation)
        volume_arr = volume.volume
        assert volume_arr.shape[2] == len(headers), \
            "Dimension mismatch - {:d} slices but {:d} headers".format(volume_arr.shape[-1], len(headers))

        # Check if dir_path exists.
        dir_path = io_utils.mkdirs(dir_path)

        num_slices = len(headers)
        filename_format = "I%0" + str(max(4, ceil(log10(num_slices)))) + "d.dcm"

        filepaths = [os.path.join(dir_path, filename_format % (s + 1)) for s in range(num_slices)]
        if self.num_workers:
            slices = [volume_arr[..., s] for s in range(num_slices)]
            with mp.Pool(self.num_workers) as p:
                out = p.starmap_async(_write_dicom_file, zip(slices, headers, filepaths))
                out.wait()
        else:
            for s in range(num_slices):
                _write_dicom_file(volume_arr[..., s], headers[s], filepaths[s])

        # Reformat image to original orientation (before saving).
        # We do this, because saving should not affect the existing state of any variable.
        volume.reformat(original_orientation)
