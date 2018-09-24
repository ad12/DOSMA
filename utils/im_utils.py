import cv2
import os
import numpy as np
import SimpleITK as sitk
from utils import io_utils

TIFF_EXTENSION = '.tiff'
TIF_EXTENSION = '.tif'

# depth first = (slices, x, y)
# else (x, y, slices)
DEPTH_FIRST = False


def write_2d(dir_path, mask):
    """Save mask to directory
    :param dir_path: path to directory
    :param mask: a 3D numpy array
    """

    depth_index = 0 if DEPTH_FIRST else 2

    dir_path = io_utils.check_dir(dir_path)
    y_true = np.squeeze(mask) * 255
    num_slices = y_true.shape[depth_index]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        im = y_true[i, :, :] if DEPTH_FIRST else y_true[:, :, i]
        cv2.imwrite(os.path.join(dir_path, slice_name), im)


def write_3d(filepath, mask):
    """ Write mask as 3d image
    :param filepath: filepath to save mask
    :param mask: a 3D numpy array

    :raises ValueError if format is not tiff
    """
    if not filepath.endswith(TIFF_EXTENSION):
        raise ValueError('Filepath must have %s extension' % TIFF_EXTENSION)

    # no permute if depth first
    permute_order = [0, 1, 2] if DEPTH_FIRST else [2, 0, 1]

    visual_mask = np.asarray(mask * 255, np.uint8)
    visual_mask = np.transpose(visual_mask, permute_order)
    img = sitk.GetImageFromArray(visual_mask)
    sitk.WriteImage(img, filepath)