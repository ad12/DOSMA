import numpy as np
import nibabel.orientations as nibo

from data_io.orientation import get_transpose_inds, get_flip_inds, __orientation_standard_to_nib__
from data_io.format_io import ImageDataFormat


class MedicalVolume():
    """Wrapper for 3D volumes """

    def __init__(self, volume: np.ndarray, pixel_spacing: tuple, orientation: tuple, scanner_origin: tuple,
                 headers=None):
        """
        :param volume: a 3D numpy array
        :param pixel_spacing: pixel/voxel spacing for volumes
        :param orientation: tuple of standardized orientation in RAS+ format
        :param scanner_origin: origin in scanner coordinate system
        """
        self._volume = volume
        self.pixel_spacing = tuple(pixel_spacing)
        self.orientation = tuple(orientation)
        self.scanner_origin = tuple(scanner_origin)
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
        # Check if new_orientation is the same as current orientation
        assert type(new_orientation) is tuple, "Orientation must be a tuple"
        if new_orientation == self.orientation:
            return

        transpose_inds = get_transpose_inds(self.orientation, new_orientation)

        volume = np.transpose(self.volume, transpose_inds)
        pixel_spacing = tuple([self.pixel_spacing[i] for i in transpose_inds])
        orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = get_flip_inds(orientation, new_orientation)
        volume = np.flip(volume, axis=flip_axs_inds)
        scanner_origin = list(self.scanner_origin)
        nib_coords = nibo.axcodes2ornt(__orientation_standard_to_nib__(orientation))

        for i in range(len(scanner_origin)):
            if i in flip_axs_inds:
                r_ind = int(nib_coords[i, 0])
                alpha_val = int(nib_coords[i,1])
                scanner_origin[r_ind] = alpha_val*pixel_spacing[i] * (volume.shape[i] - 1) + scanner_origin[r_ind]

        self._volume = volume
        self.pixel_spacing = tuple(pixel_spacing)
        self.orientation = tuple(new_orientation)
        self.scanner_origin = tuple(scanner_origin)

    def is_identical(self, mv):
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def is_same_dimensions(self, mv):
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        return mv.pixel_spacing == self.pixel_spacing and mv.orientation == self.orientation and mv.scanner_origin == self.scanner_origin and mv.volume.shape == self.volume.shape

    def match_orientation(self, mv):
        if type(mv) != type(self):
            raise TypeError('type(mv) must be %s' % str(type(self)))

        mv.reformat(self.orientation)

    def match_orientation_batch(self, mvs):
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
