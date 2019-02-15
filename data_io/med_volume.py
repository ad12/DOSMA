import numpy as np
import nibabel.orientations as nibo

from data_io import format_io
from data_io.orientation import get_transpose_inds, get_flip_inds, __orientation_standard_to_nib__
from utils import io_utils


class MedicalVolume():
    """Wrapper for 3D volumes """

    def __init__(self, volume: np.ndarray, pixel_spacing: tuple, orientation: tuple, scanner_origin: tuple):
        """
        :param volume: a 3D numpy array
        :param pixel_spacing: pixel/voxel spacing for volume
        :param orientation: tuple of standardized orientation in RAS+ format
        :param scanner_origin: origin in scanner coordinate system
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.orientation = orientation
        self.scanner_origin = scanner_origin

    def save_volume(self, filepath, data_format='nifti'):
        """
        Write volume to nifti format
        :param filepath: filepath to save data
        """
        writer = format_io.get_writer(data_format)
        writer.save(self, filepath)

    def reformat(self, new_orientation: tuple):
        # Check if new_orientation is the same as current orientation
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
                scanner_origin[r_ind] = -pixel_spacing[i] * (volume.shape[i] - 1) + scanner_origin[r_ind]

        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.orientation = new_orientation
        self.scanner_origin = scanner_origin

