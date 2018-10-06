from utils import io_utils


class MedicalVolume():
    def __init__(self, volume, pixel_spacing):
        self.volume = volume
        self.pixel_spacing = pixel_spacing

    def save_volume(self, filename):
        io_utils.save_nifti(filename, self.volume, self.pixel_spacing)
