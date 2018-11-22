from utils import io_utils


class MedicalVolume():
    """Medical volume is a 3D array with pixel spacing acquired from dicom"""

    def __init__(self, volume, pixel_spacing):
        """
        :param volume: 3D numpy array
        :param pixel_spacing: pixel spacing for 3D volume
        """
        self.volume = volume
        self.pixel_spacing = pixel_spacing

    def save_volume(self, filepath):
        """
        Write volume to nifti format
        :param filepath: filepath to save data
        """
        assert filepath.endswith('.nii.gz'), "Filepath must end in `.nii.gz` (nifti) format"

        io_utils.save_nifti(filepath, self.volume, self.pixel_spacing)
