import cv2
import os
import numpy as np
import SimpleITK as sitk

# depth first = (slices, x, y)
# else (x, y, slices)
DEPTH_FIRST = False


def check_dir(dir_path):
    """
    If directory does not exist, make directory
    :param dir_path: path to directory
    :return: path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def write_2d(dir_path, mask):
    """
    Save ground truth mask to directory
    :param dir_path: path to directory
    :param y_true: numpy array of ground truth labels
    """

    depth_index = 0 if DEPTH_FIRST else 2

    dir_path = check_dir(dir_path)
    y_true = np.squeeze(mask) * 255
    num_slices = y_true.shape[depth_index]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        im = y_true[i, :, :] if DEPTH_FIRST else y_true[:, :, i]
        cv2.imwrite(os.path.join(dir_path, slice_name), im)


def write_3d(mask_filepath, mask):
    # no permute if depth first
    permute_order = [0, 1, 2] if DEPTH_FIRST else [2, 0, 1]

    visual_mask = np.asarray(mask * 255, np.uint8)
    visual_mask = np.transpose(visual_mask, permute_order)
    img = sitk.GetImageFromArray(visual_mask)
    sitk.WriteImage(img, mask_filepath)