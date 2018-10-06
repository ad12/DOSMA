import os
import re
from abc import ABC, abstractmethod
from time import gmtime, strftime

import numpy as np
import scipy.ndimage as sni
from natsort import natsorted
from nipype.interfaces.elastix import Registration, ApplyWarp

import defaults
import file_constants as fc
from med_objects.med_volume import MedicalVolume
from utils import dicom_utils
from utils import io_utils


class ScanSequence(ABC):
    NAME = ''

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (self.NAME,
                                                                          strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        self.tissues = []
        self.dicom_path = os.path.abspath(dicom_path) if dicom_path is not None else None
        self.dicom_ext = dicom_ext

        self.volume = None
        self.ref_dicom = None

        if load_path:
            self.load_data(load_path)

        if dicom_path is not None:
            self.__load_dicom__()

    def __load_dicom__(self):
        dicom_path = self.dicom_path
        dicom_ext = self.dicom_ext

        self.volume, self.refs_dicom = dicom_utils.load_dicom(dicom_path, dicom_ext)

        self.ref_dicom = self.refs_dicom[0]

    def get_dimensions(self):
        return self.volume.volume.shape

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

        metadata = dict()
        for variable_name in self.__serializable_variables__():
            metadata[variable_name] = self.__getattribute__(variable_name)

        io_utils.save_pik(filepath, metadata)

    def load_data(self, load_dirpath):
        load_dirpath = self.__save_dir__(load_dirpath, create_dir=False)

        if not os.path.isdir(load_dirpath):
            raise NotADirectoryError('%s does not exist' % load_dirpath)

        filepath = os.path.join(load_dirpath, '%s.data' % self.NAME)

        metadata = io_utils.load_pik(filepath)
        for key in metadata.keys():
            self.__setattr__(key, metadata[key])

        self.__load_dicom__()

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

    def __serializable_variables__(self):
        return ['volume', 'dicom_path', 'dicom_ext']


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

        echo_times = []
        # get echo_time time for each refs_dicom
        for i in range(len(refs_dicom)):
            echo_time = float(refs_dicom[i].EchoTime)
            echo_times.append(echo_time)

        # rank echo times from 0 --> num_echos-1
        ordered_echo_times = natsorted(list(set(echo_times)))
        num_echo_times = len(ordered_echo_times)
        assert num_echo_times == expected_num_echos

        ordered_subvolumes_dict = dict()

        for i in range(num_echo_times):
            echo_time = ordered_echo_times[i]
            inds = np.where(np.asarray(echo_times) == echo_time)
            inds = inds[0]

            ordered_subvolumes_dict[i] = inds

        subvolumes_dict = ordered_subvolumes_dict

        # number of spin lock times should go evenly into subvolume
        if (len(refs_dicom) % num_echo_times) != 0:
            raise ValueError('Uneven number of dicom files - %d dicoms, %d spin lock times/echos' % (
                len(refs_dicom), num_echo_times))

        num_slices_per_subvolume = len(refs_dicom) / num_echo_times

        for key in subvolumes_dict.keys():
            if (len(subvolumes_dict[key]) != num_slices_per_subvolume):
                raise ValueError('subvolume \'%s\' (echo_time %s) has %d slices. Expected %d slices' % (
                    key, echo_times[key - 1], len(subvolumes_dict[key]), num_slices_per_subvolume))

        for key in subvolumes_dict.keys():
            sv = subvolumes_dict[key]
            sv = MedicalVolume(volume.volume[..., sv], volume.pixel_spacing)

            subvolumes_dict[key] = sv

        return subvolumes_dict, echo_times

    def __load_interregistered_files__(self, interregistered_dirpath):
        print('loading interregistered files')
        if 'interregistered' not in interregistered_dirpath:
            raise ValueError('Invalid path for loading %s interregistered files' % self.NAME)

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError('No interregistered files found')

        indices = []
        subvolumes = []
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError('%s is not an interregisterd \'.gz.nii\' file.' % subfile)

            subfile_num = int(subfile_nums[0])
            indices.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume = io_utils.load_nifti(filepath)

            subvolumes.append(subvolume)

        assert len(indices) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError('No interregistered files found')

        subvolumes_dict = dict()
        for i in range(len(indices)):
            subvolumes_dict[indices[i]] = subvolumes[i]

        return subvolumes_dict

    def __dilate_mask__(self, mask_path, temp_path, dil_rate=defaults.DEFAULT_MASK_DIL_RATE,
                        dil_threshold=defaults.DEFAULT_MASK_DIL_THRESHOLD):
        mask = io_utils.load_nifti(mask_path)
        mask = mask.volume
        dilated_mask = sni.gaussian_filter(np.asarray(mask.volume, dtype=np.float32),
                                           sigma=dil_rate) > dil_threshold
        fixed_mask = np.asarray(dilated_mask,
                                dtype=np.int8)
        fixed_mask_filepath = os.path.join(temp_path, 'dilated-mask.nii.gz')

        dilated_mask_volume = MedicalVolume(fixed_mask, mask.pixel_spacing)
        dilated_mask_volume.save_volume(fixed_mask_filepath)

        return fixed_mask_filepath

    def __interregister_base_file__(self, base_image_info, target_path, temp_path, mask_path=None,
                                    parameter_files=[fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]):
        base_image_path, base_time_id = base_image_info
        # Register base image to the target image
        print('Registering %s (base image)' % base_image_path)
        transformation_files = []

        use_mask_arr = [False, True]

        for i in range(len(parameter_files)):
            use_mask = use_mask_arr[i]
            pfile = parameter_files[i]
            reg = Registration()
            reg.inputs.fixed_image = target_path
            reg.inputs.moving_image = base_image_path
            reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_path,
                                                                     '%03d_param%i' % (base_time_id, i)))
            reg.inputs.parameters = pfile

            if use_mask and mask_path is not None:
                fixed_mask_filepath = self.__dilate_mask__(mask_path, temp_path)
                reg.inputs.fixed_mask = fixed_mask_filepath

            reg.terminal_output = fc.NIPYPE_LOGGING
            reg_output = reg.run()
            reg_output = reg_output.outputs
            transformation_files.append(reg_output.transform[0])

        return reg_output.warped_file, transformation_files

    def __apply_transform__(self, image_info, transformation_files, temp_path):
        filename, image_id = image_info
        print('Applying transform %s' % filename)
        warped_file = ''
        for f in transformation_files:
            reg = ApplyWarp()
            reg.inputs.moving_image = filename

            reg.inputs.transform_file = f
            reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_path,
                                                                     '%03d' % image_id))
            reg.terminal_output = fc.NIPYPE_LOGGING
            reg_output = reg.run()
            warped_file = reg_output.outputs.warped_file

        assert warped_file != ''

        return warped_file
