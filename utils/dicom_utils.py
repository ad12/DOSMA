import os

import numpy as np
import pydicom
from natsort import natsorted

from med_objects.med_volume import MedicalVolume

__VOLUME_DIMENSIONS__ = 3
__EPSILON__ = 1e-8


def get_pixel_spacing(dicom_filepath):
    """
    Get pixel spacing from dicom header
    :param dicom_filepath: Filepath to single dicom
    :return: tuple of pixel spacing (r, c, d)
    """
    ref_dicom = pydicom.read_file(dicom_filepath)

    # Load spacing values (in mm)
    pixelSpacing = (float(ref_dicom.PixelSpacing[0]),
                    float(ref_dicom.PixelSpacing[1]),
                    float(ref_dicom.SpacingBetweenSlices))

    return pixelSpacing

def load_dicom(dicom_path, dicom_ext=None):
    """Load dicoms into numpy array

    Required:
    :param dicom_path: string path to directory where dicoms are stored

    Optional:
    :param dicom_ext: string extension for dicom files. Default is None, meaning the extension will not be checked

    :return tuple of MedicalVolume, list of dicom headers

    :raises NotADirectoryError if dicom pat does not exist or is not a directory

    Note: this function sorts files using natsort, an intelligent sorting tool.
        Please label your dicoms in a sequenced manner (e.g.: dicom1,dicom2,dicom3,...)
    """
    if not os.path.isdir(dicom_path):
        raise NotADirectoryError("Directory %s does not exist" % dicom_path)

    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(dicom_path):
        for filename in fileList:
            if (dicom_ext is None) or (dicom_ext is not None and dicom_ext in filename.lower()):
                lstFilesDCM.append(os.path.join(dirName, filename))

    lstFilesDCM = natsorted(lstFilesDCM)
    if len(lstFilesDCM) == 0:
        raise FileNotFoundError("No files found in directory %s" % dicom_path)

    # Get reference file
    ref_dicom = pydicom.read_file(lstFilesDCM[0])

    # Load dimensions based on rows, columns, and slices along Z axis
    pixelDims = (int(ref_dicom.Rows), int(ref_dicom.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    pixelSpacing = (float(ref_dicom.PixelSpacing[0]),
                    float(ref_dicom.PixelSpacing[1]),
                    float(ref_dicom.SpacingBetweenSlices))

    dicom_array = np.zeros(pixelDims, dtype=ref_dicom.pixel_array.dtype)

    refs_dicom = []
    for dicom_filename in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(dicom_filename, force=True)
        refs_dicom.append(ds)
        dicom_array[:, :, lstFilesDCM.index(dicom_filename)] = ds.pixel_array

    volume = MedicalVolume(dicom_array, pixelSpacing)

    return volume, refs_dicom


def whiten_volume(x):
    """Whiten volume by mean and std of all pixels
    :param x: 3D numpy array (MRI volume)
    :rtype: whitened 3D numpy array
    """
    if len(x.shape) != __VOLUME_DIMENSIONS__:
        raise ValueError("Dimension Error: input has %d dimensions. Expected %d" % (x.ndims, __VOLUME_DIMENSIONS__))

    # Add epsilon to avoid dividing by 0
    return (x - np.mean(x)) / (np.std(x) + __EPSILON__)
