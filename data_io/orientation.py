"""
Standardized orientation among different libraries

Various libraries have different ways to interpret information read from various image formats (Dicom, NIfTi, etc)
Here, we standardize the orientation for this framework.

We indicate how to translate reading/writing orientations from current libraries (Nibabel, PyDicom, etc)

Orientation Conventions:
-------------------------
- All orientations are in patient voxel coordinates - i.e. (i, j, k) --> voxel at numpy array position [i, j, k]
- Left: corresponds to patient (not observer) left, RIGHT: corresponds to patient (not observer) right

Standard Orientation format:
----------------------------
All directions point to the increasing direction - i.e. from -x to x
- "LR": left to right; "RL": right to left
- "PA": posterior to anterior; "AP": anterior to posterior
- "IS": inferior to superior; "SI": superior to inferior

@author: Arjun Desai
        (C) Stanford University, 2019
"""

# Default Orientations
SAGITTAL = ('SI', 'AP', 'LR')

__EXPECTED_ORIENTATION_TUPLE_LEN__ = 3
__SUPPORTED_ORIENTATIONS__ = ['LR', 'RL', 'PA', 'AP', 'IS', 'SI']
__ORIENTATIONS_TO_AXIS_ID__ = {'LR': 0, 'RL': 0,
                               'PA': 1, 'AP': 1,
                               'IS': 2, 'SI': 2}


def __check_orientation__(orientation: tuple):
    is_orientation_format = len(orientation) == __EXPECTED_ORIENTATION_TUPLE_LEN__ and sum(
        [type(o) is str for o in orientation]) == __EXPECTED_ORIENTATION_TUPLE_LEN__

    orientation_str_exists = sum(
        [o in __SUPPORTED_ORIENTATIONS__ for o in orientation]) == __EXPECTED_ORIENTATION_TUPLE_LEN__

    orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in orientation]
    unique_ids = len(orientation_ids) == len(set(orientation_ids))

    if not is_orientation_format or not orientation_str_exists or not unique_ids:
        raise ValueError(
            "Orientation format mismatch: Orientations must be tuple of strings of length %d." % __EXPECTED_ORIENTATION_TUPLE_LEN__)


def get_transpose_inds(curr_orientation: tuple, new_orientation: tuple):
    __check_orientation__(curr_orientation)
    __check_orientation__(new_orientation)

    curr_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in curr_orientation]
    new_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in new_orientation]

    if set(curr_orientation_ids) != set(new_orientation_ids):
        raise ValueError("Orientation mismatch: Both curr_orientation and new_orientation must contain the same axes")

    transpose_inds = [curr_orientation_ids.index(n_o) for n_o in new_orientation_ids]

    return tuple(transpose_inds)


def get_flip_inds(curr_orientation: tuple, new_orientation: tuple):
    __check_orientation__(curr_orientation)
    __check_orientation__(new_orientation)

    curr_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in curr_orientation]
    new_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in new_orientation]

    if curr_orientation_ids != new_orientation_ids:
        raise ValueError('All axis orientations (S/I, L/R, A/P) must be in the same location in tuple')

    flip_axs_inds = []
    for i in range(__EXPECTED_ORIENTATION_TUPLE_LEN__):
        if curr_orientation[i] != new_orientation[i]:
            flip_axs_inds.append(i)

    return flip_axs_inds


# Nibabel to standard orientation conversion utils
nib_to_standard_orientation_map = {'R': 'LR', 'L': 'RL',
                                   'A': 'PA', 'P': 'AP',
                                   'S': 'IS', 'I': 'SI'}


def __orientation_nib_to_standard__(nib_orientation):
    """
    Convert Nibabel orientation to the standard orientation
    :param nib_orientation:
    :return:
    """
    orientation = []
    for symb in nib_orientation:
        orientation.append(nib_to_standard_orientation_map[symb])
    return tuple(orientation)


def __orientation_standard_to_nib__(orientation):
    """
    Convert standard orientation to Nibabel orientation
    :param orientation:
    :return:
    """
    nib_orientation = []
    for symb in orientation:
        nib_orientation.append(symb[1])
    return tuple(nib_orientation)
