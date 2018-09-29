from abc import ABC, abstractmethod
from utils import dicom_utils
import numpy as np
from utils import io_utils
import os
import file_constants as fc
from time import gmtime, strftime
from natsort import natsorted
import re


class ScanSequence(ABC):
    NAME = ''

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (self.NAME,
                                                                          strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        self.tissues = []
        self.dicom_path = dicom_path
        self.dicom_ext = dicom_ext

        self.volume = None
        self.pixel_spacing = None
        self.ref_dicom = None

        if load_path:
            self.load_data(load_path)

        if dicom_path is not None:
            self.__load_dicom__()

    def __load_dicom__(self):
        dicom_path = self.dicom_path
        dicom_ext = self.dicom_ext

        self.volume, self.refs_dicom, self.pixel_spacing = dicom_utils.load_dicom(dicom_path, dicom_ext)

        self.ref_dicom = self.refs_dicom[0]

    def get_dimensions(self):
        return self.volume.shape

    def __add_tissue__(self, new_tissue):
        contains_tissue = any([tissue.ID == new_tissue.ID for tissue in self.tissues])
        if contains_tissue:
            raise ValueError('Tissue already exists')

        self.tissues.append(new_tissue)

    def __data_filename__(self):
        return '%s.%s' % (self.NAME, io_utils.DATA_EXT)

    def save_data(self, save_dirpath):
        # write other data as ref
        save_dirpath = self.__save_dir__(save_dirpath)
        filepath = os.path.join(save_dirpath, '%s.data' % self.NAME)
        data = {'volume': self.volume, 'pixel_spacing': self.pixel_spacing}

        io_utils.save_pik(filepath, data)

    def load_data(self, load_dirpath):
        load_dirpath = self.__save_dir__(load_dirpath, create_dir=False)

        if not os.path.isdir(load_dirpath):
            raise NotADirectoryError('%s does not exist' % load_dirpath)

        filepath = os.path.join(load_dirpath, '%s.data' % self.NAME)
        data = io_utils.load_pik(filepath)

        #self.ref_dicom = data['ref_dicom']
        self.volume = data['volume']
        self.pixel_spacing = data['pixel_spacing']

    def __save_dir__(self, dirpath, create_dir=True):
        name_len = len(self.NAME) + 2  # buffer
        if self.NAME in dirpath[-name_len:]:
            scan_dirpath = os.path.join(dirpath, '%s_data' % self.NAME)
        else:
            scan_dirpath = dirpath

        scan_dirpath = os.path.join(scan_dirpath, '%s_data' % self.NAME)

        if create_dir:
            scan_dirpath = io_utils.check_dir(scan_dirpath)

        return scan_dirpath


class TargetSequence(ScanSequence):

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

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError('No interregistered files found')

        spin_lock_times = []
        subvolumes = []
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError('%s is not an interregisterd \'.gz.nii\' file.' % subfile)

            subfile_num = float(subfile_nums[0])
            spin_lock_times.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume_arr, self.pixel_spacing = io_utils.load_nifti(filepath)
            subvolumes.append(subvolume_arr)

        assert len(spin_lock_times) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError('No interregistered files found')

        subvolumes_dict = dict()
        for i in range(len(spin_lock_times)):
            subvolumes_dict[spin_lock_times[i]] = subvolumes[i]

        return subvolumes_dict





