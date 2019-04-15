"""
MedicalVolume: Wrapper for 3D volumes

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import nibabel as nib
import numpy as np

from data_io import orientation as stdo
from data_io.format_io import ImageDataFormat
from data_io.orientation import get_transpose_inds, get_flip_inds
from defaults import SCANNER_ORIGIN_DECIMAL_PRECISION
from copy import deepcopy

class MedicalVolume():
    """Wrapper for 3D volumes """

    def __init__(self, volume: np.ndarray, affine: np.ndarray, headers=None):
        """
        :param volume: a 3D numpy array
        :param affine: a 4x4 numpy array resembling affine matrix transform in RAS+ coordinates
        """
        self._volume = volume
        self._affine = np.array(affine)
        self._headers = headers

    def save_volume(self, filepath, data_format: ImageDataFormat = ImageDataFormat.nifti):
        """
        Write volumes to specified image data format
        :param filepath: filepath to save data
        :param data_format: an ImageDataFormat
        """
        import data_io.format_io_utils
        writer = data_io.format_io_utils.get_writer(data_format)

        writer.save(self, filepath)

    def reformat(self, new_orientation: tuple):
        """
        Reorients self to a specified orientation
        :param new_orientation: a tuple specifying orientation

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
        """
        # Check if new_orientation is the same as current orientation
        assert type(new_orientation) is tuple, "Orientation must be a tuple"
        if new_orientation == self.orientation:
            return

        temp_orientation = self.orientation
        temp_affine = np.array(self._affine)

        transpose_inds = get_transpose_inds(temp_orientation, new_orientation)

        volume = np.transpose(self.volume, transpose_inds)
        for i in range(len(transpose_inds)):
            temp_affine[..., i] = self._affine[..., transpose_inds[i]]

        temp_orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = list(get_flip_inds(temp_orientation, new_orientation))

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
        """
        Check if another medical volume is identical to self
        Two volumes are identical if they have the same pixel_spacing, orientation, scanner_origin, and volume
        :param mv: a MedicalVolume
        :return: a boolean
        """
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def is_same_dimensions(self, mv):
        """
        Check if two volumes have the same dimensions
        Two volumes have the same dimensions if they have the same pixel_spacing, orientation, and scanner_origin
        :param mv:
        :return: a boolean
        """
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        return mv.pixel_spacing == self.pixel_spacing and mv.orientation == self.orientation and mv.scanner_origin == self.scanner_origin and mv.volume.shape == self.volume.shape

    def match_orientation(self, mv):
        """
        Reorient another MedicalVolume to orientation specified by self.orientation
        :param mv: a MedicalVolume
        """
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        mv.reformat(self.orientation)

    def match_orientation_batch(self, mvs):
        """
        Reorient a group of MedicalVolumes to orientation specified by self.orientation
        :param mvs: a list/tuple of MedicalVolumes
        """
        for mv in mvs:
            self.match_orientation(mv)

    # Properties
    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value: np.ndarray):
        assert value.ndim == 3, "Volume must be 3D"

        # if the volume is of a different shape, the headers are no longer valid, so delete
        # all reorientations are done as part of MedicalVolume, so reorientations are permitted
        # however, external setting of the volume to a different shape array is not allowed
        if self._volume.shape != value.shape:
            self._headers = None

        self._volume = value

    @property
    def headers(self):
        return self._headers

    @property
    def pixel_spacing(self):
        """
        Get pixel spacing in order of current orientation
        :return a tuple
        """
        vecs = self._affine[:3, :3]
        ps = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

        assert len(ps) == 3, "pixel spacing must have length of 3"
        return ps

    @property
    def orientation(self):
        """
        Get the closest orientation in standard orientation coordinates
        :return: a tuple of standard orientation coordinates (see orientation.py for more information on format)
        """
        nib_orientation = nib.aff2axcodes(self._affine)
        return stdo.__orientation_nib_to_standard__(nib_orientation)

    @property
    def scanner_origin(self):
        """
        Get the scanner origin in global RAS+ x,y,z coordinates
        :return:
        """
        return tuple(self._affine[:3, 3])

    @property
    def affine(self):
        return self._affine
