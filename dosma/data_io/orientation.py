"""Standardized orientation convention and utilities.

Medical image orientation convention is library and image format (DICOM, NIfTI, etc.)
dependent and is often difficult to interpret. This makes it challenging to intelligently
and rapidly reformat images.

We adopt a easily interpretable orientation representation
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
:class:`dosma.data_io.MedicalVolume`.

"""

__all__ = [
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
