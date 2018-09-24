from abc import ABC, abstractmethod
from utils import dicom_utils
import numpy as np
from utils import io_utils
import os
import file_constants as fc
from time import gmtime, strftime
from natsort import natsorted

class ScanSequence(ABC):
    NAME = ''

    def __init__(self, dicom_path=None, dicom_ext=None):
        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (self.NAME, strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        self.tissues = []
        self.dicom_path = dicom_path
        self.dicom_ext = dicom_ext

        if dicom_path is not None:
            self.__load_dicom__()

    def __load_dicom__(self):
        dicom_path = self.dicom_path
        dicom_ext = self.dicom_ext

        self.volume, self.refs_dicom = dicom_utils.load_dicom(dicom_path, dicom_ext)

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
    def interregister(self, target):
        """
        Register this scan to the target scan
        :param target: a 3D numpy volume that will serve as the base for registration
                        (should be segmented slices if multiecho)
        :return:
        """
        pass

    def __split_volumes__(self, expected_num_echos):
        refs_dicom = self.refs_dicom
        volume = self.volume
        subvolumes_dict = dict()

        # get echo_time time for each refs_dicom
        for i in range(len(refs_dicom)):
            echo_time = refs_dicom[i].EchoTime
            if echo_time in subvolumes_dict.keys():
                subvolumes_dict[echo_time].append(i)
            else:
                subvolumes_dict[echo_time] = [i]

        # assert expected number of echos
        num_echo_times = len(list(subvolumes_dict.keys()))
        assert num_echo_times == expected_num_echos

        # number of spin lock times should go evenly into subvolume
        if (len(refs_dicom) % num_echo_times) != 0:
            raise ValueError('Uneven number of dicom files - %d dicoms, %d spin lock times/echos' % (len(refs_dicom),
                                                                                                     num_echo_times))

        num_slices_per_subvolume = len(refs_dicom) / num_echo_times

        for key in subvolumes_dict.keys():
            if (len(subvolumes_dict[key]) != num_slices_per_subvolume):
                raise ValueError('echo_time \'%s\' has %d slices. Expected %d slices' % (key, len(subvolumes_dict[key]), num_slices_per_subvolume))

        for key in subvolumes_dict.keys():
            sv = subvolumes_dict[key]
            sv = volume[..., sv]
            subvolumes_dict[key] = sv

        return subvolumes_dict

    def __load_interregistered_files__(self, interregistered_dirpath):
        if 'interregistered' not in interregistered_dirpath:
            raise ValueError('Invalid path for loading %s interregistered files' % self.NAME)

        subdirs = io_utils.get_subdirs(interregistered_dirpath)
        subdirs = natsorted(subdirs)

        spin_lock_times = []
        subvolumes = []
        for subdir in subdirs:
            spin_lock_times.append(float(subdir))
            filepath = os.path.join(interregistered_dirpath, subdir, 'output.nii')
            subvolume_arr = io_utils.load_nifti(filepath)
            subvolumes.append(subvolume_arr)

        assert len(spin_lock_times) == len(subvolumes), "Number of subvolumes mismatch"

        subvolumes_dict = dict()
        for i in range(len(spin_lock_times)):
            subvolumes_dict[spin_lock_times[i]] = subvolumes[i]

        return subvolumes_dict





