import os
from abc import ABC, abstractmethod
from enum import Enum

from data_io.med_volume import MedicalVolume
from utils import io_utils


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

    def __init__(self, volumetric_map=None):
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

    def save_data(self, dirpath):
        if self.volumetric_map is not None:
            self.volumetric_map.save_volume(os.path.join(dirpath, self.NAME, '%s.nii.gz' % self.NAME))

        for volume_name in self.additional_volumes.keys():
            self.additional_volumes[volume_name].save_volume(os.path.join(dirpath, self.NAME, '%s-%s.nii.gz' %
                                                                          (self.NAME, volume_name)))

    def load_data(self, dirpath):
        self.volumetric_map = io_utils.load_nifti(os.path.join(dirpath, self.NAME, '%s.nii.gz' % self.NAME))

    def add_additional_volume(self, name, volume):
        if not isinstance(volume, MedicalVolume):
            raise TypeError('`volume` must be of type MedicalVolume')
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


if __name__ == '__main__':
    print(type(QuantitativeValues.T1_RHO.name))
    print(QuantitativeValues.T1_RHO.value == 1)
