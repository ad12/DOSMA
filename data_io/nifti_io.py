"""
File detailing modules for NIfTI format IO

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import os

import nibabel as nib
import numpy as np

from data_io.format_io import DataReader, DataWriter, ImageDataFormat
from data_io.med_volume import MedicalVolume
from defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION
from utils import io_utils

__NIFTI_EXTENSIONS__ = ('.nii', '.nii.gz')


def contains_nifti_extension(a_str: str):
    """
    Check if a string ends with one of the accepted nifti extensions
    :param a_str: a string
    :return: a boolean
    """
    bool_list = [a_str.endswith(ext) for ext in __NIFTI_EXTENSIONS__]
    return bool(sum(bool_list))


class NiftiReader(DataReader):
    """
    A class for reading data in NIfTI format
    """
    data_format_code = ImageDataFormat.nifti

    def __normalize_affine(self, affine):
        # determine vector for through-plane pixel direction (k)
        # 1. Normalize k_vector by magnitude
        # 2. Multiply by magnitude given by SpacingBetweenSlices field
        # These actions are done to avoid rounding errors that might result from float subtraction

        aff = np.array(affine)
        i_vec = np.round(np.array(aff[:3, 0]), AFFINE_DECIMAL_PRECISION)
        j_vec = np.round(np.array(aff[:3, 1]), AFFINE_DECIMAL_PRECISION)
        k_vec = np.round(np.array(aff[:3, 2]), AFFINE_DECIMAL_PRECISION)

        aff[:3, 0] = i_vec
        aff[:3, 1] = j_vec
        aff[:3, 2] = k_vec

        aff[:3, 3] = np.round(aff[:3, 3], SCANNER_ORIGIN_DECIMAL_PRECISION)

        return aff

    def load(self, filepath):
        """
        Load image data from filepath

        A NIfTI file should only correspond to one volume. As a result, only one volume is outputted

        :param filepath: a string defining filepath to nifti file

        :raises FileNotFoundError if filepath not found
        :raises ValueError if filepath does not contain supported NIfTI extension
        :return: a MedicalVolume
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError('%s not found' % filepath)

        if not contains_nifti_extension(filepath):
            raise ValueError('%s must be a file with extension `.nii` or `.nii.gz`' % filepath)

        nib_img = nib.load(filepath)
        nib_img_affine = nib_img.affine
        nib_img_affine = self.__normalize_affine(nib_img_affine)

        np_img = nib_img.get_fdata()

        return MedicalVolume(np_img, nib_img_affine)


class NiftiWriter(DataWriter):
    """
    A class for writing data in NIfTI format
    """
    data_format_code = ImageDataFormat.nifti

    def save(self, im: MedicalVolume, filepath: str):
        """
        Save a MedicalVolume in NIfTI format
        :param im: a Medical Volume
        :param filepath: a string defining filepath to save image to

        :raises ValueError if filepath does not contain supported NIfTI extension
        """
        if not contains_nifti_extension(filepath):
            raise ValueError('%s must be a file with extension `.nii` or `.nii.gz`' % filepath)

        # Create dir if does not exist
        io_utils.check_dir(os.path.dirname(filepath))

        nib_affine = im.affine
        np_im = im.volume
        nib_img = nib.Nifti1Image(np_im, nib_affine)

        nib.save(nib_img, filepath)


if __name__ == '__main__':
    import scipy.io as sio

    load_filepath = '../dicoms/mapss_eg/multi-echo/gt-%d.nii.gz'
    save_path = '../dicoms/mapss_eg/matfiles-nii'
    nr = NiftiReader()
    for i in range(7):
        med_vol = nr.load(load_filepath % i)
        sio.savemat(os.path.join(save_path, '%d.mat' % i), {'data': med_vol.volume})
