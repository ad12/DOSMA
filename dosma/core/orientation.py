"""Standardized orientation convention and utilities.

Medical image orientation convention is library and image format (DICOM, NIfTI, etc.)
dependent and is often difficult to interpret. This makes it challenging to intelligently
and rapidly reformat images.

We adopt a human readable orientation representation
for the dimensions and define utilities to convert between different orientation formats
from current libraries (Nibabel, PyDicom, ITK, etc).

Orientations are represented by string axis codes:

- ``"LR"``: left to right; ``"RL"``: right to left
- ``"PA"``: posterior to anterior; ``"AP"``: anterior to posterior
- ``"IS"``: inferior to superior; ``"SI"``: superior to inferior

A :class:`MedicalVolume` object with orientation ``("SI", "AP", "LR")`` has an
array where the first dimension spans superior -> inferior, the second dimension
spans anterior -> posterior, and the third dimension spans left -> right. Voxel
at (i,j,k) index ``(0,0,0)`` would be the (superior, anterior, left) corner.

In many cases, images are not acquired in the standard plane convention, but rather
in a rotated frame. In this case, the orientations correspond to the closest axis
the a particular dimension.

Two general conventions are followed:

- All orientations are in patient voxel coordinates. Image data from (i, j, k)
  corresponds to the voxel at array position ``arr[i,j,k]``.
- Left: corresponds to patient (not observer) left,
  right: corresponds to patient (not observer) right.

We adopt the RAS+ standard (as defined by NIfTI) for orienting our images.
The ``+`` in RAS+ indicates that all directions point to the increasing direction.
i.e. from -x to x:.

Image spacing, direction, and global origin are represented by a 4x4 affine matrix (:math:`A`) and
is identical to the nibabel affine matrix
(see `nibabel <https://nipy.org/nibabel/coordinate_systems.html>`_).
The affine matrix converts pixel coordinates (i, j, k) into world (NIfTI) coordinates (x, y, z).

.. math::

    \\begin{bmatrix} x\\\\y\\\\z\\\\1\\end{bmatrix} = A
    \\begin{bmatrix} i\\\\j\\\\k\\\\1\\end{bmatrix}


For example,

.. math::

    \\begin{bmatrix} x\\\\y\\\\z\\\\1 \\end{bmatrix} =
    \\begin{bmatrix} 0 & 0 & 1.5 & -61.6697\\\\-0.3125 & 0 & 0 & 50.8516\\\\
    0 & -0.3125 & 0 & 88.5876\\\\0 & 0 & 0 & 1 \\end{bmatrix}
    \\begin{bmatrix} i\\\\j\\\\k\\\\1\\end{bmatrix}

For details on how the affine matrix is used for reformatting see
:class:`dosma.core.MedicalVolume`.

"""
from typing import Sequence, Union

import nibabel.orientations as nibo
import numpy as np

__all__ = [
    "to_affine",
    "get_transpose_inds",
    "get_flip_inds",
    "orientation_nib_to_standard",
    "orientation_standard_to_nib",
    "SAGITTAL",
    "CORONAL",
    "AXIAL",
]


SAGITTAL = ("SI", "AP", "LR")
CORONAL = ("SI", "LR", "AP")
AXIAL = ("AP", "LR", "SI")

__EXPECTED_ORIENTATION_TUPLE_LEN__ = 3
__SUPPORTED_ORIENTATIONS__ = ["LR", "RL", "PA", "AP", "IS", "SI"]
__ORIENTATIONS_TO_AXIS_ID__ = {"LR": 0, "RL": 0, "PA": 1, "AP": 1, "IS": 2, "SI": 2}


def __check_orientation__(orientation: tuple):
    """Check if orientation tuple defines a valid orientation.

    Args:
        orientation (tuple[str]): Image orientation in standard orientation format.

    Raises:
        ValueError: If orientation tuple is invalid.
    """
    is_orientation_format = (
        len(orientation) == __EXPECTED_ORIENTATION_TUPLE_LEN__
        and sum([type(o) is str for o in orientation]) == __EXPECTED_ORIENTATION_TUPLE_LEN__
    )

    orientation_str_exists = (
        sum([o in __SUPPORTED_ORIENTATIONS__ for o in orientation])
        == __EXPECTED_ORIENTATION_TUPLE_LEN__
    )

    orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in orientation]
    unique_ids = len(orientation_ids) == len(set(orientation_ids))

    if not is_orientation_format or not orientation_str_exists or not unique_ids:
        raise ValueError(
            "Orientation format mismatch: Orientations must be tuple of strings of "
            "length {}".format(__EXPECTED_ORIENTATION_TUPLE_LEN__)
        )


def get_transpose_inds(curr_orientation: tuple, new_orientation: tuple):
    """Get indices for reordering planes from ``curr_orientation`` to ``new_orientation``.

    Only permuted order of reformatting the image planes is returned.
    For example, ``("SI", "AP", "LR")`` and ``("IS", "PA", "RL")`` will have no permuted
    indices because "SI"/"IS", "AP"/"PA" and "RL"/"LR" each correspond to the same
    plane.

    Args:
        curr_orientation (tuple[str]): Current image orientation.
        new_orientation (tuple[str]): New image orientation.

    Returns:
        tuple[int]: Axes to transpose to change orientation.

    Examples:
        >>> get_transpose_inds(("SI", "AP", "LR"), ("AP", "SI", "LR"))
        (1,0,2)
        >>> get_transpose_inds(("SI", "AP", "LR"), ("IS", "PA", "RL"))
        (0,1,2)
    """
    __check_orientation__(curr_orientation)
    __check_orientation__(new_orientation)

    curr_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in curr_orientation]
    new_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in new_orientation]

    if set(curr_orientation_ids) != set(new_orientation_ids):
        raise ValueError(
            "Orientation mismatch: Both curr_orientation and new_orientation "
            "must contain the same axes"
        )

    transpose_inds = [curr_orientation_ids.index(n_o) for n_o in new_orientation_ids]

    return tuple(transpose_inds)


def get_flip_inds(curr_orientation: tuple, new_orientation: tuple):
    """Get indices to flip from ``curr_orientation`` to ``new_orientation``.

    Args:
        curr_orientation (tuple[str]): Current image orientation.
        new_orientation (tuple[str]): New image orientation.

    Returns:
        list[int]: Axes to flip.

    Raises:
        ValueError: If mismatch in orientation indices. To avoid this error,
            use :func:`get_transpose_inds` prior to flipping.

    Examples:
        >>> get_transpose_inds(("SI", "AP", "LR"), ("IS", "AP", "RL"))
        [0, 2]
    """
    __check_orientation__(curr_orientation)
    __check_orientation__(new_orientation)

    curr_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in curr_orientation]
    new_orientation_ids = [__ORIENTATIONS_TO_AXIS_ID__[o] for o in new_orientation]

    if curr_orientation_ids != new_orientation_ids:
        raise ValueError(
            "All axis orientations (S/I, L/R, A/P) must be ordered. "
            "Use `get_transpose_inds` to reorder axes."
        )

    flip_axs_inds = []
    for i in range(__EXPECTED_ORIENTATION_TUPLE_LEN__):
        if curr_orientation[i] != new_orientation[i]:
            flip_axs_inds.append(i)

    return flip_axs_inds


# Nibabel to standard orientation conversion utils.
__nib_to_standard_orientation_map__ = {
    "R": "LR",
    "L": "RL",
    "A": "PA",
    "P": "AP",
    "S": "IS",
    "I": "SI",
}


def orientation_nib_to_standard(nib_orientation):
    """Convert Nibabel orientation to the standard dosma orientation format.

    Args:
        nib_orientation: a RAS+ tuple orientation used by Nibabel.

    Returns:
        tuple[str]: Image orientation in the standard orientation format.

    Examples:
        >>> orientation_nib_to_standard(("R", "A", "S"))
        ("LR", "PA", "IS")
    """
    orientation = []
    for symb in nib_orientation:
        orientation.append(__nib_to_standard_orientation_map__[symb])
    return tuple(orientation)


def orientation_standard_to_nib(orientation):
    """Convert standard dosma orientation format to Nibabel orientation.

    Args:
        orientation: Image orientation in the standard orientation format.

    Returns:
        tuple[str]: RAS+ tuple orientation used by Nibabel.

    Examples:
        >>> orientation_nib_to_standard(("LR", "PA", "IS"))
        ("R", "A", "S")
    """
    nib_orientation = []
    for symb in orientation:
        nib_orientation.append(symb[1])
    return tuple(nib_orientation)


def to_affine(
    orientation,
    spacing: Sequence[Union[int, float]] = None,
    origin: Sequence[Union[int, float]] = None,
):
    """Convert orientation, spacing, and origin data into affine matrix.

    Args:
        orientation (Sequence[str]): Image orientation in the standard orientation format
            (e.g. ``("LR", "AP", "SI")``).
        spacing (int(s) | float(s)): Number(s) corresponding to pixel spacing of each direction.
            If a single value, same pixel spacing is used for all directions.
            If sequence is less than length of ``orientation``, remaining direction have unit
            spacing (i.e. ``1``). Defaults to unit spacing ``(1, 1, 1)``
        origin (int(s) | float(s)): The ``(x0, y0, z0)`` origin for the scan.
            If a single value, same origin is used for all directions.
            If sequence is less than length of ``orientation``, remaining direction have standard
            origin (i.e. ``0``). Defaults to ``(0, 0, 0)``

    Returns:
        ndarray: A 4x4 ndarray representing the affine matrix.

    Examples:
        >>> to_affine(("SI", "AP", "RL"), spacing=(0.5, 0.5, 1.5), origin=(10, 20, 0))
        array([[-0. , -0. , -1.5,  10. ],
               [-0. , -0.5, -0. ,  20. ],
               [-0.5, -0. , -0. ,  30. ],
               [ 0. ,  0. ,  0. ,   1. ]])

    Note:
        This method assumes all direction follow the standard principal directions in the normative
        patient orientation. Moving along one direction of the array only moves along one fo the
        normative directions.
    """

    def _format_numbers(input, default_val, name, expected_num):
        """Formats (sequence of) numbers (spacing, origin) into standard 3-length tuple."""
        if input is None:
            return (default_val,) * expected_num
        if isinstance(input, (int, float)):
            return (input,) * expected_num

        if not isinstance(input, (np.ndarray, Sequence)) or len(input) > expected_num:
            raise ValueError(
                f"`{name}` must be a real number or sequence (length<={expected_num}) "
                f"of real numbers. Got {input}"
            )
        input = tuple(input)

        if len(input) < expected_num:
            input += (default_val,) * (expected_num - len(input))
        assert len(input) == expected_num

        return input

    if len(orientation) == 2:
        orientation = _infer_orientation(orientation)
    __check_orientation__(orientation)
    spacing = _format_numbers(spacing, 1, "spacing", len(orientation))
    origin = _format_numbers(origin, 0, "origin", len(orientation))

    affine = np.eye(4)
    start_ornt = nibo.io_orientation(affine)
    end_ornt = nibo.axcodes2ornt(orientation_standard_to_nib(orientation))
    ornt = nibo.ornt_transform(start_ornt, end_ornt)

    transpose_idxs = ornt[:, 0].astype(int)
    flip_idxs = ornt[:, 1]

    affine[:3] = affine[:3][transpose_idxs]
    affine[:3] *= flip_idxs[..., np.newaxis]
    affine[:3, :3] *= np.asarray(spacing)
    affine[:3, 3] = np.asarray(origin)

    return affine


def _infer_orientation(orientation):
    """Infer 3-length orientation from 2-length orientation.

    Args:
        orientation: The incomplete orientation.

    Returns:
        tuple[str, str, str]: Standard orientation.
    """
    idxs = {__ORIENTATIONS_TO_AXIS_ID__[k] for k in orientation}
    if len(orientation) != 2 or len(idxs) != 2:
        raise ValueError(
            "`orientation` must be an incomplete orientation that encodes orthogonal directions"
        )

    missing_ornt = [k for k, v in __ORIENTATIONS_TO_AXIS_ID__.items() if v not in idxs][0]
    return tuple(orientation) + (missing_ornt,)
