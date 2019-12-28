"""MedicalVolume

This module defines `MedicalVolume`, which is a wrapper for 3D volumes.
"""

import nibabel as nib
import numpy as np

import pydicom

from dosma.data_io import orientation as stdo
from dosma.data_io.format_io import ImageDataFormat
from dosma.defaults import SCANNER_ORIGIN_DECIMAL_PRECISION

from typing import List

__all__ = ["MedicalVolume"]


class MedicalVolume(object):
    """Wrapper for 3D medical volumes.

    Medical volumes are 3D matrices representing medical data. These volumes have inherent metadata, such as pixel/voxel
        spacing, global coordinates, rotation information, which can be characterized by an affine matrix.

    Args:
        volume (np.ndarray): 3D volume.
        affine (np.ndarray): 4x4 array corresponding to affine matrix transform in RAS+ coordinates.
        headers (list[pydicom.FileDataset]): Headers for DICOM files.
    """

    def __init__(self, volume: np.ndarray, affine: np.ndarray, headers: List[pydicom.FileDataset] = None):
        if headers and len(headers) != volume.shape[-1]:
            raise ValueError("Header mismatch. {:d} headers, but {:d} slices".format(len(headers), volume.shape[-1]))

        self._volume = volume
        self._affine = np.array(affine)
        self._headers = headers

    def save_volume(self, file_path: str, data_format: ImageDataFormat = ImageDataFormat.nifti):
        """Write volumes in specified data format.

        Args:
            file_path (str): File path to save data. May be modified to follow convention given by the data format in which
                the volume will be saved.
            data_format (ImageDataFormat): Format to save data.
        """
        import dosma.data_io.format_io_utils
        writer = dosma.data_io.format_io_utils.get_writer(data_format)

        writer.save(self, file_path)

    def reformat(self, new_orientation: tuple):
        """Reorients volume to a specified orientation.

        Reorientation method:
        ---------------------
        - Axis transpose and flipping are linear operations and therefore can be treated independently
        - working example: ('AP', 'SI', 'LR') --> ('RL', 'PA', 'SI')
        1. Transpose volume and RAS orientation to appropriate column in matrix
            eg. ('AP', 'SI', 'LR') --> ('LR', 'AP', 'SI') - transpose_inds=[2, 0, 1]
        2. Flip volume across corresponding axes
            eg. ('LR', 'AP', 'SI') --> ('RL', 'PA', 'SI') - flip axes 0,1

        Reorientation method implementation:
        ------------------------------------
        1. Transpose: Switching (transposing) axes in volume is the same as switching columns in affine matrix
        2. Flipping: Negate each column corresponding to pixel axis to flip (i, j, k) and reestablish origins based on
                     flipped axes

        Args:
            new_orientation (tuple): New orientation.
        """
        # Check if new_orientation is the same as current orientation
        assert type(new_orientation) is tuple, "Orientation must be a tuple"
        if new_orientation == self.orientation:
            return

        temp_orientation = self.orientation
        temp_affine = np.array(self._affine)

        transpose_inds = stdo.get_transpose_inds(temp_orientation, new_orientation)

        volume = np.transpose(self.volume, transpose_inds)
        for i in range(len(transpose_inds)):
            temp_affine[..., i] = self._affine[..., transpose_inds[i]]

        temp_orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = list(stdo.get_flip_inds(temp_orientation, new_orientation))

        volume = np.flip(volume, axis=flip_axs_inds)

        a_vecs = temp_affine[:3, :3]
        a_origin = temp_affine[:3, 3]

        # phi is a vector of 1s and -1s, where 1 indicates no flip, and -1 indicates flip
        # phi is used to determine which columns in affine matrix to flip
        phi = np.ones([1, len(a_origin)]).flatten()
        phi[flip_axs_inds] *= -1

        b_vecs = np.array(a_vecs)
        for i in range(len(phi)):
            b_vecs[:, i] *= phi[i]

        # get number of pixels to shift by on each axis (should be 0 when not flipping - i.e. phi<0 mask)
        vol_shape_vec = ((np.asarray(volume.shape) - 1) * (phi < 0).astype(np.float32)).transpose()
        b_origin = np.round(a_origin.flatten() - np.matmul(b_vecs, vol_shape_vec).flatten(),
                            SCANNER_ORIGIN_DECIMAL_PRECISION)

        temp_affine = np.array(self.affine)
        temp_affine[:3, :3] = b_vecs
        temp_affine[:3, 3] = b_origin
        temp_affine[temp_affine == 0] = 0  # get rid of negative 0s

        self._affine = temp_affine

        assert self.orientation == new_orientation, "Orientation mismatch: Expected: %s. Got %s" % (
            str(self.orientation),
            str(new_orientation))
        self._volume = volume

    def is_identical(self, mv):
        """Check if another medical volume is identical.

        Two volumes are identical if they have the same pixel_spacing, orientation, scanner_origin, and volume.

        Args:
            mv (MedicalVolume): Volume to compare with.

        Returns:
            bool: `True` if identical, `False` otherwise.
        """
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def __allclose_spacing(self, mv, precision: int = None):
        """Check if spacing between self and another medical volume is within tolerance.

        Tolerance is `10 ** (-precision)`.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal. If not specified, check that
                affine matrices between two volumes are identical. Defaults to `None`.

        Returns:
            bool: `True` if spacing between two volumes within tolerance, `False` otherwise.
        """
        if precision:
            tol = 10 ** (-precision)
            return np.allclose(mv.affine[:3, :3], self.affine[:3, :3], atol=tol) and np.allclose(mv.scanner_origin,
                                                                                                 self.scanner_origin,
                                                                                                 rtol=tol)
        else:
            return (mv.affine == self.affine).all()

    def is_same_dimensions(self, mv, precision: int = None):
        """Check if two volumes have the same dimensions.

        Two volumes have the same dimensions if they have the same pixel_spacing, orientation, and scanner_origin.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal. If not specified, check that
                affine matrices between two volumes are identical. Defaults to `None`.

        Returns:
            bool: `True` if pixel spacing, orientation, and scanner origin between two volumes within tolerance, `False`
                otherwise.

        Raises:
            TypeError: If `mv` is not a MedicalVolume.
        """
        if type(mv) != type(self):
            raise TypeError("'mv' must be of type {}".format(str(type(self))))

        return self.__allclose_spacing(mv, precision) \
               and mv.orientation == self.orientation \
               and mv.volume.shape == self.volume.shape

    def match_orientation(self, mv):
        """Reorient another MedicalVolume to orientation specified by self.orientation.

        Args:
            mv (MedicalVolume): Volume to reorient.
        """
        if type(mv) != type(self):
            raise TypeError("'mv' must be of type {}".format(str(type(self))))

        mv.reformat(self.orientation)

    def match_orientation_batch(self, mvs):
        """Reorient a collection of MedicalVolumes to orientation specified by self.orientation.

        Args:
            mvs (list[MedicalVolume]): Collection of MedicalVolumes.
        """
        for mv in mvs:
            self.match_orientation(mv)

    @property
    def volume(self):
        """np.ndarray: 3D numpy array representing volume values."""
        return self._volume

    @volume.setter
    def volume(self, value: np.ndarray):
        """
        If the volume is of a different shape, the headers are no longer valid, so delete all reorientations are done
            as part of MedicalVolume, so reorientations are permitted.

        However, external setting of the volume to a different shape array is not allowed.
        """
        assert value.ndim == 3, "Volume must be 3D"

        if self._volume.shape != value.shape:
            self._headers = None

        self._volume = value

    @property
    def headers(self):
        """list[pydicom.FileDataset]: Headers for DICOM files."""
        return self._headers

    @property
    def pixel_spacing(self):
        """tuple[float]: Pixel spacing in order of current orientation."""
        vecs = self._affine[:3, :3]
        ps = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

        assert len(ps) == 3, "Pixel spacing must have length of 3"
        return ps

    @property
    def orientation(self):
        """tuple[str]: Image orientation in standard orientation format.

        See orientation.py for more information on conventions.
        """
        nib_orientation = nib.aff2axcodes(self._affine)
        return stdo.orientation_nib_to_standard(nib_orientation)

    @property
    def scanner_origin(self):
        """tuple[float]: Scanner origin in global RAS+ x,y,z coordinates.
        """
        return tuple(self._affine[:3, 3])

    @property
    def affine(self):
        """np.ndarray: 4x4 affine matrix for volume in current orientation."""
        return self._affine
