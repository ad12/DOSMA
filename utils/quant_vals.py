import os
from abc import ABC, abstractmethod
from enum import Enum

from data_io import format_io_utils as fio_utils
from data_io.format_io import ImageDataFormat
from data_io.med_volume import MedicalVolume
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT

__all__ = ['QuantitativeValues', 'QuantitativeValue', 'T1Rho', 'T2', 'T2Star']


class QuantitativeValues(Enum):
    """Enum of quantitative values that can be analyzed"""
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


class QuantitativeValue(ABC):
    ID = 0
    NAME = ''

    @staticmethod
    def get_qv(qv_id):
        """Find QuantitativeValue enum using id
        :param qv_id: either enum number or name in lower case
        :return: Quantitative value corresponding to id

        :raise ValueError: if no QuantitativeValue corresponding to id found
        """
        for qv in [T1Rho(), T2(), T2Star()]:
            if qv.NAME.lower() == qv_id or qv.NAME == qv_id:
                return qv

        raise ValueError('Quantitative Value with name or id %s not found' % str(qv_id))

    @staticmethod
    def load_qvs(dirpath):
        qvs = []
        for qv in [T1Rho(), T2(), T2Star()]:
            possible_qv_filepath = os.path.join(dirpath, qv.NAME, '%s.nii.gz' % qv.NAME)
            if os.path.isfile(possible_qv_filepath):
                qv.load_data(dirpath)
                qvs.append(qv)

        return qvs

    @staticmethod
    def save_qvs(dirpath, qvs):
        for qv in qvs:
            if not isinstance(qv, QuantitativeValue):
                raise TypeError('All members of `qvs` must be instances of QuantitativeValue')
            qv.save_data(dirpath)

    def __init__(self, volumetric_map: MedicalVolume=None):
        # Main 3D quantitative value map (MedicalVolume)
        if volumetric_map is not None and not isinstance(volumetric_map, MedicalVolume):
            raise TypeError('`volumetric_map` must be of type MedicalVolume')

        self.volumetric_map = volumetric_map

        # Sometimes there are additional volumes that we want to keep track of
        # i.e. r2 values, goodness of fit, etc
        # Store these as string_name: MedicalVolume (i.e. "r2: MedicalVolume)
        # To add values to this field, use the add_additional_volume method below
        # these results will not be loaded
        self.additional_volumes = dict()

    def save_data(self, dirpath, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        """

        :param dirpath:
        :param data_format:
        :return:
        """
        if data_format != ImageDataFormat.nifti:
            import warnings
            warnings.warn(
                "Due to bit depth issues, only nifti format is supported for quantitative values. Writing as nifti file...")
            data_format = ImageDataFormat.nifti

        if self.volumetric_map is not None:
            filepath = os.path.join(dirpath, self.NAME, '%s.nii.gz' % self.NAME)
            # filepath = fio_utils.convert_format_filename(filepath, data_format)
            self.volumetric_map.save_volume(filepath, data_format=data_format)

        for volume_name in self.additional_volumes.keys():
            add_vol_filepath = os.path.join(dirpath, self.NAME, '%s-%s.nii.gz' % (self.NAME, volume_name))
            # add_vol_filepath = fio_utils.convert_format_filename(add_vol_filepath, data_format)
            self.additional_volumes[volume_name].save_volume(add_vol_filepath, data_format=data_format)

    def load_data(self, dirpath):
        filepath = os.path.join(dirpath, self.NAME, '%s.nii.gz' % self.NAME)
        qv_volume = fio_utils.generic_load(filepath, expected_num_volumes=1)
        self.volumetric_map = qv_volume

    def add_additional_volume(self, name, volume):
        if not isinstance(volume, MedicalVolume):
            raise TypeError('`volumes` must be of type MedicalVolume')
        self.additional_volumes[name] = volume

    @abstractmethod
    def get_enum(self):
        pass


class T1Rho(QuantitativeValue):
    ID = 1
    NAME = 't1_rho'

    def get_enum(self):
        return QuantitativeValues.T1_RHO


class T2(QuantitativeValue):
    ID = 2
    NAME = 't2'

    def get_enum(self):
        return QuantitativeValues.T2


class T2Star(QuantitativeValue):
    ID = 3
    NAME = 't2_star'

    def get_enum(self):
        return QuantitativeValues.T2_STAR
