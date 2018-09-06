import os
import numpy as np
from natsort import natsorted
import pydicom
import math
from pydicom.tag import Tag


__VOLUME_DIMENSIONS__ = 3
__EPSILON__ = 1e-8


def load_dicom(dicom_path, dicom_ext=None):
    """Load dicoms into numpy array

    Required:
    :param dicom_path: string path to directory where dicoms are stored

    Optional:
    :param dicom_ext: string extension for dicom files. Default is None, meaning the extension will not be checked

    :rtype: 3D numpy array

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
    pixelSpacing = (float(ref_dicom.PixelSpacing[0]), float(ref_dicom.PixelSpacing[1]), float(ref_dicom.SliceThickness))

    x = np.arange(0.0, (pixelDims[0] + 1) * pixelSpacing[0], pixelSpacing[0])
    y = np.arange(0.0, (pixelDims[1] + 1) * pixelSpacing[1], pixelSpacing[1])
    z = np.arange(0.0, (pixelDims[2] + 1) * pixelSpacing[2], pixelSpacing[2])

    dicom_array = np.zeros(pixelDims, dtype=ref_dicom.pixel_array.dtype)

    for dicom_filename in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(dicom_filename, force=True)
        dicom_array[:, :, lstFilesDCM.index(dicom_filename)] = ds.pixel_array

    return dicom_array, ref_dicom


def whiten_volume(x):
    """Whiten volume by mean and std of all pixels
    :param x: 3D numpy array (MRI volume)
    :rtype: whitened 3D numpy array
    """
    if len(x.shape) != __VOLUME_DIMENSIONS__:
        raise ValueError("Dimension Error: input has %d dimensions. Expected %d" % (x.ndims, __VOLUME_DIMENSIONS__))

    # Add epsilon to avoid dividing by 0
    return (x - np.mean(x)) / (np.std(x) + __EPSILON__)


def split_volume(volume, echos=1):
    """Split volume of multiple echos into multiple subvolumes of same echo

    Required:
    :param volume: 3D numpy array of muliple-echos

    Optional:
    :param echos: The number of echos in the volume

    :rtype: list of numpy arrays (subvolumes)

    :raises ValueError if volume is not 3D, echos <= 0, or not equal number of slices per echo

    Usage:
    DESS has 2 echos - "split_volume(volume, 2)" --> [echo1, echo2]
    """
    if (len(volume.shape) != __VOLUME_DIMENSIONS__):
        raise ValueError("Dimension Error: input has %d dimensions. Expected %d" % (volume.ndims, __VOLUME_DIMENSIONS__))
    if (echos <= 0):
        raise ValueError('There must be at least 1 echo per volume')

    depth = volume.shape[2]
    if (depth % echos != 0):
        raise ValueError('Number of slices per echo must be the same')

    sub_volumes = []
    for i in range(echos):
        sub_volumes.append(volume[:, :, i::echos])

    return sub_volumes