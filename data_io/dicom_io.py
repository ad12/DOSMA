"""
File detailing modules for Dicom format IO

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import os
from math import log10, ceil

import nibabel as nib
import numpy as np
import pydicom
from natsort import natsorted

from data_io import orientation as stdo
from data_io.format_io import DataReader, DataWriter, ImageDataFormat
from data_io.med_volume import MedicalVolume
from defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION
from utils import io_utils

__DICOM_EXTENSIONS__ = ('.dcm',)
TOTAL_NUM_ECHOS_KEY = (0x19, 0x107e)


def contains_dicom_extension(a_str: str):
    """
    Check if a string ends with one of the accepted dicom extensions
    :param a_str: a string
    :return: a boolean
    """
    bool_list = [a_str.endswith(ext) for ext in __DICOM_EXTENSIONS__]
    return bool(sum(bool_list))


def __update_np_dtype__(np_array, bit_depth):
    """
    Return copy of np_array with bit-depth and type specified here
    Try to use float whenever possible - if float does not capture range, try to use
    Note pydicom only supports writing dicoms with bit-depth 8/16 - only supports bit depths 8/16
    :param np_array: a Numpy array
    :param bit_depth: the bit depth to
    :return: a copy of input np_array with dtype
    """
    assert bit_depth in [8, 16], "Only bit-depths of 8 and 16 are currently supported."
    dtype_dict = {8: [(np.int8, -128, 127), (np.uint8, 0, 255)],
                  16: [(np.float16, -6.55e4, 6.55e4 - 1), (np.int16, -2 ** 15, 2 ** 15), (np.uint16, 0, 2 ** 16 - 1)]}
    supported_floats = [np.float16]
    curr_min = np.min(np_array)
    curr_max = np.max(np_array)
    contains_float = (np_array % 1 != 0).any()

    dtypes = dtype_dict[bit_depth]

    new_dtype = None
    for dtype, dtype_min, dtype_max in dtypes:
        if curr_min < dtype_min or curr_max > dtype_max:
            continue
        new_dtype = dtype
        break
    if not new_dtype:
        raise ValueError('Cannot cast numpy array (%s) to bit-depth of %d bits' % (str(np_array.dtype), bit_depth))

    if contains_float and new_dtype not in supported_floats:
        raise TypeError('Array contains float. Cannot cast to numpy array (%s) to %s' % (str(np_array.dtype),
                                                                                         new_dtype))

    return np_array.astype(new_dtype)


def LPSplus_to_RASplus(headers):
    """
    Convert from LPS+ orientation (default for Dicom) to RAS+ standardized orientation
    :param headers: a list of FileDatasets
    :return: a tuple of orientation and scanner origin
    """
    im_dir = headers[0].ImageOrientationPatient
    in_plane_pixel_spacing = headers[0].PixelSpacing

    orientation = np.zeros([3, 3])

    # determine vector for in-plane pixel directions (i, j)
    i_vec, j_vec = np.asarray(im_dir[:3]).astype(np.float64), np.asarray(im_dir[3:]).astype(
        np.float64)  # unique to pydicom, please revise if using different library to load dicoms
    i_vec, j_vec = np.round(i_vec, AFFINE_DECIMAL_PRECISION), np.round(j_vec, AFFINE_DECIMAL_PRECISION)
    i_vec = i_vec * in_plane_pixel_spacing[0]
    j_vec = j_vec * in_plane_pixel_spacing[1]

    # determine vector for through-plane pixel direction (k)
    # 1. Normalize k_vector by magnitude
    # 2. Multiply by magnitude given by SpacingBetweenSlices field
    # These actions are done to avoid rounding errors that might result from float subtraction
    k_vec = np.asarray(headers[1].ImagePositionPatient).astype(np.float64) - np.asarray(
        headers[0].ImagePositionPatient).astype(np.float64)
    k_vec = np.round(k_vec, AFFINE_DECIMAL_PRECISION)
    # k_vec_magnitude = np.sqrt(np.sum(k_vec**2))
    # assert k_vec_magnitude == headers[0].SpacingBetweenSlices
    # k_vec = k_vec / k_vec_magnitude * headers[0].SpacingBetweenSlices

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


class DicomReader(DataReader):
    """
    A class for reading dicom files

    Key Notes:
    ------------------
    1. Dicom utilizes LPS convention:
        - LPS: right --> left, anterior --> posterior, inferior --> superior
        - we will call it LPS+, such that letters correspond to increasing end of axis
    """
    data_format_code = ImageDataFormat.dicom

    def load(self, dicom_dirpath):
        """Load dicoms into numpy array

        Required:
        :param dicom_dirpath: string path to directory where dicoms are stored

        :return list of MedicalVolumes

        :raises NotADirectoryError if dicom pat does not exist or is not a directory
        :raises FileNotFoundError if no dicom files found in directory

        Note: This function sorts files using natsort, an intelligent sorting tool.
              Please label your dicoms in a sequenced manner (e.g.: dicom1,dicom2,dicom3,...)
        """
        if not os.path.isdir(dicom_dirpath):
            raise NotADirectoryError("Directory %s does not exist" % dicom_dirpath)

        possible_files = os.listdir(dicom_dirpath)

        lstFilesDCM = []
        for f in possible_files:
            if contains_dicom_extension(f):
                lstFilesDCM.append(os.path.join(dicom_dirpath, f))

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No files found in directory %s" % dicom_dirpath)

        # Get reference file
        ref_dicom = pydicom.read_file(lstFilesDCM[0])

        dicom_data = {}

        for dicom_filename in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(dicom_filename, force=True)
            echo_number = ds.EchoNumbers
            echo_ind = echo_number - 1

            if echo_ind not in dicom_data.keys():
                dicom_data[echo_ind] = {'headers': [], 'arr': []}

            dicom_data[echo_ind]['headers'].append(ds)
            dicom_data[echo_ind]['arr'].append(ds.pixel_array)

        vols = []
        for k in sorted(list(dicom_data.keys())):
            dd = dicom_data[k]
            headers = dd['headers']
            if len(headers) == 0:
                continue
            arr = np.stack(dd['arr'], axis=-1)

            affine = LPSplus_to_RASplus(headers)

            vol = MedicalVolume(arr,
                                affine,
                                headers=headers)
            vols.append(vol)

        return vols


class DicomWriter(DataWriter):
    """
    A class for writing data in dicom format
    """
    data_format_code = ImageDataFormat.dicom

    def __write_dicom_file__(self, np_slice: np.ndarray, header: pydicom.FileDataset, filepath: str):
        """
        Replace data in header with 2D numpy array and write to filepath
        :param np_slice: a 2D numpy array
        :param header: a pydicom.FileDataset with fields populated
        :param filepath: Filepath to write dicom to
        """
        expected_dimensions = header.Rows, header.Columns
        assert np_slice.shape == expected_dimensions, "In-plane dimension mismatch - expected shape %s, got %s" % (
            str(expected_dimensions),
            str(np_slice.shape))

        np_slice_bytes = np_slice.tobytes()
        bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)
        if bit_depth != header.BitsAllocated:
            np_slice = __update_np_dtype__(np_slice, header.BitsAllocated)
            np_slice_bytes = np_slice.tobytes()
            bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)

        assert bit_depth == header.BitsAllocated, "Bit depth mismatch: Expected %d got %d" % (header.BitsAllocated,
                                                                                              bit_depth)

        header.PixelData = np_slice_bytes

        header.save_as(filepath)

    def save(self, im, filepath):
        """
        Save a medical volume in dicom format
        :param im: a Medical Volume
        :param filepath: a path to a directory to store dicom files

        :raises ValueError if im (MedicalVolume) does not have initialized headers
        :raises ValueError if im was flipped across any axis. Flipping changes scanner origin, which is currently not handled
        """
        # Get orientation indicated by headers
        headers = im.headers
        if headers is None:
            raise ValueError('MedicalVolume headers must be initialized to save as a dicom')

        affine = LPSplus_to_RASplus(headers)
        orientation = stdo.__orientation_nib_to_standard__(nib.aff2axcodes(affine))

        # Currently do not support mismatch in scanner_origin
        if tuple(affine[:3, 3]) != im.scanner_origin:
            raise ValueError(
                'Scanner origin mismatch. Currently we do not handle mismatch in scanner origin (i.e. cannot flip across axis)')

        # reformat medical volume to expected orientation specified by dicom headers
        # store original orientation so we can undo the dicom-specific reformatting
        original_orientation = im.orientation

        im.reformat(orientation)
        volume = im.volume
        assert volume.shape[2] == len(headers), "Dimension mismatch - %d slices but %d headers" % (
            volume.shape[-1], len(headers))

        # check if filepath exists
        filepath = io_utils.check_dir(filepath)

        num_slices = len(headers)
        filename_format = 'I%0' + str(max(4, ceil(log10(num_slices)))) + 'd.dcm'

        for s in range(num_slices):
            s_filepath = os.path.join(filepath, filename_format % (s + 1))
            self.__write_dicom_file__(volume[..., s], headers[s], s_filepath)

        # reformat image to original orientation (before saving)
        # we do this, because saving should not affect the existing state of any variable
        im.reformat(original_orientation)


if __name__ == '__main__':
    dicom_filepath = '../dicoms/mapss_eg/'
    save_path = '../dicoms/mapss_eg/multi-echo-write/'
    r = DicomReader()
    A = r.load(dicom_filepath)
    from data_io.nifti_io import NiftiWriter

    w = NiftiWriter()
    for i in range(len(A)):
        w.save(A[i], os.path.join(save_path, '%d.nii.gz' % i))
