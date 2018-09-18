from abc import ABC, abstractmethod
from utils import dicom_utils
import numpy as np
from utils import io_utils


class ScanSequence(ABC):
    NAME = ''
    def __init__(self, dicom_path, dicom_ext=None):
        self.tissues = []
        self.dicom_path = dicom_path
        self.dicom_ext = dicom_ext

        self.__load_dicom__()

    def __load_dicom__(self):
        dicom_path = self.dicom_path
        dicom_ext = self.dicom_ext

        self.volume, self.ref_dicom = dicom_utils.load_dicom(dicom_path, dicom_ext)

    def get_dimensions(self):
        return self.volume.shape

    def __add_tissue__(self, new_tissue):
        contains_tissue = any([tissue.ID == new_tissue.ID for tissue in self.tissues])
        if contains_tissue:
            raise ValueError('Tissue already exists')

        self.tissues.append(new_tissue)

    def __data_filename__(self):
        return '%s.%s' % (self.NAME, io_utils.DATA_EXT)

    @abstractmethod
    def save_data(self, save_dirpath):
        pass

    @abstractmethod
    def load_data(self, load_dirpath):
        pass


class TargetSequence(ScanSequence):

    def __init__(self, dicom_path, dicom_ext):
        super().__init__(dicom_path, dicom_ext)

    def preprocess_volume(self, volume):
        """
        Preprocess segmentation volume by whitening the volume
        :param volume: segmentation volume
        :return:
        """
        return dicom_utils.whiten_volume(volume)

    @abstractmethod
    def segment(self, model, tissue):
        """
        Segment based on model
        :param model: a SegModel instance
        :return: a 3D numpy binary array of segmentation
        """
        pass


class NonTargetSequence(ScanSequence):

    @abstractmethod
    def intraregister(self, subvolumes):
        """
        Register subvolumes to each other
        :param subvolumes:
        :return:
        """
        pass

    @abstractmethod
    def interregister(self, target):
        """
        Register this scan to the target scan
        :param target: a 3D numpy volume that will serve as the base for registration
        :return:
        """
        pass





