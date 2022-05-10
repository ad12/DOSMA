from typing import List, Tuple, Sequence

import numpy as np
import pydicom

from dosma.core import orientation as stdo
from dosma.defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION

__all__ = ["to_RAS_affine"]


def to_RAS_affine(headers: List[pydicom.FileDataset], default_ornt: Tuple[str, str] = None):
    """Convert from LPS+ orientation (default for DICOM) to RAS+ standardized orientation.

    Args:
        headers (list[pydicom.FileDataset]): Headers for DICOM files to reorient.
            Files should correspond to single volume.

    Returns:
        np.ndarray: Affine matrix.
    """
    try:
        im_dir = headers[0].ImageOrientationPatient
    except AttributeError:
        im_dir = _decode_inplane_direction(headers, default_ornt=default_ornt)
        if im_dir is None:
            raise RuntimeError("Could not determine in-plane directions from headers.")
    try:
        in_plane_pixel_spacing = headers[0].PixelSpacing
    except AttributeError:
        try:
            in_plane_pixel_spacing = headers[0].ImagerPixelSpacing
        except AttributeError:
            raise RuntimeError(
                "Could not determine in-plane pixel spacing from headers. "
                "Neither attribute 'PixelSpacing' nor 'ImagerPixelSpacing' found."
            )

    orientation = np.zeros([3, 3])

    # Determine vector for in-plane pixel directions (i, j).
    i_vec, j_vec = (
        np.asarray(im_dir[:3]).astype(np.float64),
        np.asarray(im_dir[3:]).astype(np.float64),
    )  # unique to pydicom, please revise if using different library to load dicoms
    i_vec, j_vec = (
        np.round(i_vec, AFFINE_DECIMAL_PRECISION),
        np.round(j_vec, AFFINE_DECIMAL_PRECISION),
    )
    i_vec = i_vec * in_plane_pixel_spacing[0]
    j_vec = j_vec * in_plane_pixel_spacing[1]

    # Determine vector for through-plane pixel direction (k).
    # Compute difference in patient position between consecutive headers.
    # This is the preferred method to determine the k vector.
    # If single header, take cross product between i/j vectors.
    # These actions are done to avoid rounding errors that might result from float subtraction.
    if len(headers) > 1:
        k_vec = np.asarray(headers[1].ImagePositionPatient).astype(np.float64) - np.asarray(
            headers[0].ImagePositionPatient
        ).astype(np.float64)
    else:
        slice_thickness = headers[0].get("SliceThickness", 1.0)
        i_norm = 1 / np.linalg.norm(i_vec) * i_vec
        j_norm = 1 / np.linalg.norm(j_vec) * j_vec
        k_norm = np.cross(i_norm, j_norm)
        k_vec = k_norm / np.linalg.norm(k_norm) * slice_thickness
        if hasattr(headers[0], "SpacingBetweenSlices") and headers[0].SpacingBetweenSlices < 0:
            k_vec *= -1
    k_vec = np.round(k_vec, AFFINE_DECIMAL_PRECISION)

    orientation[:3, :3] = np.stack([j_vec, i_vec, k_vec], axis=1)
    scanner_origin = headers[0].get("ImagePositionPatient", np.zeros((3,)))
    scanner_origin = np.asarray(scanner_origin).astype(np.float64)
    scanner_origin = np.round(scanner_origin, SCANNER_ORIGIN_DECIMAL_PRECISION)

    affine = np.zeros([4, 4])
    affine[:3, :3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2, :] = -1 * affine[:2, :]
    affine[3, 3] = 1

    affine[affine == 0] = 0

    return affine


def _decode_inplane_direction(headers: Sequence[pydicom.FileDataset], default_ornt=None):
    """Helper function to decode in-plane direction from header(s).

    Recall the direction in dicoms are in cartesian order ``(x,y)``,
    but numpy/dosma are in matrix order ``(y,x)``. When adding new
    methods, make sure to account for this.

    Returns:
        np.ndarray: 6-element LPS direction array where first 3 elements define
            direction for x-direction (columns) and second 3 elements define
            direction for y-direction (rows)
    """
    _patient_ornt_to_nib = {"H": "S", "F": "I"}

    if (
        len(headers) == 1
        and hasattr(headers[0], "PatientOrientation")
        and headers[0].PatientOrientation
    ):
        # Decoder: patient orientation.
        # Patient orientation is only decoded along principal direction (e.g. "FR" -> "F").
        ornt = [
            _patient_ornt_to_nib.get(k[:1], k[:1]) for k in headers[0].PatientOrientation
        ]  # (x,y)
        ornt = stdo.orientation_nib_to_standard(ornt)
        affine = stdo.to_affine(ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    if default_ornt:
        affine = stdo.to_affine(default_ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    return None