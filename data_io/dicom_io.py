import os

import nibabel as nib
import numpy as np
import pydicom
from natsort import natsorted

from data_io import orientation as stdo
from data_io.format_io import DataReader, DataWriter, ImageDataFormat
from data_io.med_volume import MedicalVolume
from math import log10, ceil

from utils import io_utils

__DICOM_EXTENSIONS__ = ('.dcm')
TOTAL_NUM_ECHOS_KEY = (0x19, 0x107e)


def contains_dicom_extension(a_str: str):
    bool_list = [a_str.endswith(ext) for ext in __DICOM_EXTENSIONS__]
    return bool(sum(bool_list))


def LPSplus_to_RASplus(headers):
    im_dir = headers[0].ImageOrientationPatient
    orientation = np.zeros([3, 3])
    i_vec, j_vec = im_dir[3:], im_dir[:3]  # unique to pydicom, please revise if using different library to load dicoms
    k_vec = np.asarray(headers[-1].ImagePositionPatient) - np.asarray(headers[0].ImagePositionPatient)
    k_vec = k_vec * np.dot(headers[0].ImagePositionPatient, np.cross(i_vec, j_vec))
    # k_div = np.abs(k_vec)
    # k_div[k_div == 0] += 1
    # orientation[:, 2] = k_vec / k_div
    orientation[:3, :3] = np.stack([i_vec, j_vec, k_vec], axis=1)
    scanner_origin = headers[0].ImagePositionPatient

    affine = np.zeros([4, 4])
    affine[:3, :3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2, :] = -1 * affine[:2, :]
    affine[3, 3] = 1

    affine[affine == 0] = 0
    nib_axcodes = nib.aff2axcodes(affine)
    std_orientation = stdo.__orientation_nib_to_standard__(nib_axcodes)

    return std_orientation, affine[:3, 3]


class DicomReader(DataReader):
    """
    Key Notes:
    ------------------
    1. Dicom utilizes LPS convention:
        - LPS: right --> left, anterior --> posterior, inferior --> superior
        - we will call it LPS+, such that letters correspond to increasing end of axis
    """
    data_format_code = ImageDataFormat.nifti

    def load(self, dicom_dirpath):
        """Load dicoms into numpy array

        Required:
        :param dicom_dirpath: string path to directory where dicoms are stored

        Optional:
        :param dicom_ext: string extension for dicom files. Default is None, meaning the extension will not be checked

        :return tuple of MedicalVolume, list of dicom headers

        :raises NotADirectoryError if dicom pat does not exist or is not a directory

        Note: this function sorts files using natsort, an intelligent sorting tool.
            Please label your dicoms in a sequenced manner (e.g.: dicom1,dicom2,dicom3,...)
        """
        if not os.path.isdir(dicom_dirpath):
            raise NotADirectoryError("Directory %s does not exist" % dicom_dirpath)

        lstFilesDCM = []
        for dirName, subdirList, fileList in os.walk(dicom_dirpath):
            for filename in fileList:
                if contains_dicom_extension(filename):
                    lstFilesDCM.append(os.path.join(dirName, filename))

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No files found in directory %s" % dicom_dirpath)

        # Get reference file
        ref_dicom = pydicom.read_file(lstFilesDCM[0])
        max_num_echos = ref_dicom[TOTAL_NUM_ECHOS_KEY].value

        # Load spacing values (in mm)
        pixelSpacing = (float(ref_dicom.PixelSpacing[0]),
                        float(ref_dicom.PixelSpacing[1]),
                        float(ref_dicom.SpacingBetweenSlices))

        dicom_data = []
        for i in range(max_num_echos):
            dicom_data.append({'headers': [], 'arr': []})

        for dicom_filename in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(dicom_filename, force=True)
            echo_number = ds.EchoNumbers
            dicom_data[echo_number - 1]['headers'].append(ds)
            dicom_data[echo_number - 1]['arr'].append(ds.pixel_array)

        vols = []
        for dd in dicom_data:
            headers = dd['headers']
            if len(headers) == 0:
                continue
            arr = np.stack(dd['arr'], axis=-1)

            rasplus_orientation, rasplus_origin = LPSplus_to_RASplus(headers)

            vol = MedicalVolume(arr,
                                pixel_spacing=pixelSpacing,
                                orientation=rasplus_orientation,
                                scanner_origin=rasplus_origin,
                                headers=headers)
            vols.append(vol)

        return vols


class DicomWriter(DataWriter):
    data_format_code = ImageDataFormat.nifti

    def __write_dicom_file__(self, np_slice: np.ndarray, header: pydicom.FileDataset, filepath: str):
        expected_dimensions = header.Rows, header.Columns
        assert np_slice.shape == expected_dimensions, "In-plane dimension mismatch - expected shape %s, got %s" % (str(expected_dimensions),
                                                                                                                   str(np_slice.shape))

        header.PixelData = np_slice.tobytes()
        header.save_as(filepath)

    def save(self, im, filepath):
        # Get orientation indicated by headers
        headers = im.headers
        if headers is None:
            raise ValueError('MedicalVolume headers must be initialized to save as a dicom')

        orientation, scanner_origin = LPSplus_to_RASplus(headers)

        # Currently do not support mismatch in scanner_origin
        if tuple(scanner_origin) != im.scanner_origin:
            raise ValueError('Scanner origin mismatch. Currently we do not handle mismatch in scanner origin (i.e. cannot flip across axis)')

        # reformat medical volume to expected orientation specified by dicom headers
        # store original orientation so we can undo the dicom-specific reformatting
        original_orientation = im.orientation

        im.reformat(orientation)
        volume = im.volume
        assert volume.shape[2] == len(headers), "Dimension mismatch - %d slices but %d headers" % (volume.shape[-1], len(headers))

        # check if filepath exists
        filepath = io_utils.check_dir(filepath)

        num_slices = len(headers)
        filename_format = 'I%0' + str(max(4, ceil(log10(num_slices)))) + 'd.dcm'

        for s in range(num_slices):
            s_filepath = os.path.join(filepath, filename_format % (s+1))
            self.__write_dicom_file__(volume[..., s], headers[s], s_filepath)

        # reformat image to original orientation (before saving)
        # we do this, because saving should not affect the existing state of any variable
        im.reformat(original_orientation)

if __name__ == '__main__':
    dicom_filepath = '../dicoms/healthy07/007'
    save_path = '../dicoms/healthy07/dcm-nifti.nii.gz'
    r = DicomReader()
    A = r.load(dicom_filepath)
    from data_io.nifti_io import NiftiWriter

    r = NiftiWriter()
    r.save(A[0], save_path)
