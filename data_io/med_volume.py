import numpy as np

from data_io.format_io import SUPPORTED_FORMATS
from data_io.orientation import get_transpose_inds
from utils import io_utils


class MedicalVolume():
    """Medical volume is a 3D array with pixel spacing acquired from dicom"""

    def __init__(self, volume: np.ndarray, pixel_spacing: tuple, orientation: tuple, origin: np.ndarray):
        """
        :param volume: 3D numpy array
        :param pixel_spacing: pixel spacing for 3D volume
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.orientation = orientation
        self.origin = origin

    def save_volume(self, filepath, format='nifti'):
        """
        Write volume to nifti format
        :param filepath: filepath to save data
        """

        if format not in SUPPORTED_FORMATS:
            raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

        assert filepath.endswith('.nii.gz'), "Filepath must end in `.nii.gz` (nifti) format"
        io_utils.save_nifti(filepath, self.volume, self.pixel_spacing)

    def reformat(self, new_orientation: tuple):
        # Check if new_orientation is the same as current orientation
        if new_orientation == self.orientation:
            return

        transpose_inds = get_transpose_inds(self.orientation, new_orientation)

        self.volume = np.transpose(self.volume, transpose_inds)
        self.pixel_spacing = tuple([self.pixel_spacing[i] for i in transpose_inds])
        self.orientation = tuple([self.orientation[i] for i in transpose_inds])
        self.origin = [self.origin[i] for i in transpose_inds]
