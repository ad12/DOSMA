"""
Abstract classes defining implementation of scan
All scan types/protocols should inherit from these abstract classes

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import os
import re
import warnings
from abc import ABC, abstractmethod
from time import localtime, strftime

import numpy as np
import scipy.ndimage as sni
from natsort import natsorted
from nipype.interfaces.elastix import Registration, ApplyWarp

import defaults
import file_constants as fc
from data_io import format_io_utils as fio_utils
from data_io.dicom_io import DicomReader
from data_io.format_io import ImageDataFormat
from data_io.med_volume import MedicalVolume
from data_io.nifti_io import NiftiReader
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from models.model import SegModel
from tissues.tissue import Tissue
from utils import io_utils


class ScanSequence(ABC):
    NAME = ''

    def __init__(self, dicom_path: str = None, load_path: str = None):
        """
        :param dicom_path: a path to the folder containing all dicoms for this scan
        :param dicom_ext: extension of these dicom files
        :param load_path: base path were data is stored

        In practice, only either dicom_path or load_path should be specified
        If both are specified, the dicom_path is used, and all information in load_path is ignored
        """
        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (self.NAME,
                                                                          strftime("%Y-%m-%d-%H-%M-%S", localtime()))
        self.tissues = []
        self.dicom_path = os.path.abspath(dicom_path) if dicom_path is not None else None

        self.volumes = None
        self.ref_dicom = None
        self.series_number = None

        # check if dicom path exists
        if (dicom_path is not None) and (not os.path.isdir(dicom_path)):
            if load_path is not None:
                warnings.warn('Dicom_path %s not found. Will load data from %s' % (dicom_path, load_path))
            else:
                raise NotADirectoryError('%s is not a directory' % dicom_path)

        # Only use dicoms if the path exists and path contains files ending in dicom_ext
        is_dicom_available = (dicom_path is not None) and (os.path.isdir(dicom_path))

        # Only load data if dicom path is not given or doesn't exist, else assume user wants to rewrite information
        if load_path and not is_dicom_available:
            self.load_data(load_path)

        if is_dicom_available:
            self.__load_dicom__()

    @abstractmethod
    def __validate_scan__(self) -> bool:
        """Validate this scan (usually done by checking dicom header tags, if available)
        :return a boolean
        """
        pass

    def __load_dicom__(self):
        """
        Store the dicoms to MedicalVolume and store a list of the dicom headers
        """
        dicom_path = self.dicom_path

        if dicom_path is None or not os.path.isdir(dicom_path):
            raise NotADirectoryError('%s not found' % dicom_path)

        dr = DicomReader()

        self.volumes = dr.load(dicom_path)
        self.ref_dicom = self.volumes[0].headers[0]

        self.__set_series_number__(self.ref_dicom.SeriesNumber)

    def __set_series_number__(self, sn: int):
        if self.series_number is not None:
            assert self.series_number == sn, "Series numbers must be identical if loading the same scan"
            return
        else:
            self.series_number = sn

    def get_dimensions(self):
        """
        Get shape of volumes
        :return: tuple of ints
        """
        return self.volumes[0].volume.shape

    def __add_tissue__(self, new_tissue: Tissue):
        """Add a tissue to the list of tissues associated with this scan
        :param new_tissue: a Tissue instance
        :raise ValueError: tissue type already exists in list
                                for example, we cannot add FemoralCartilage twice to the list of tissues
        """
        contains_tissue = any([tissue.ID == new_tissue.ID for tissue in self.tissues])
        if contains_tissue:
            raise ValueError('Tissue already exists')

        self.tissues.append(new_tissue)

    def __data_filename__(self):
        """Get filename for storing serialized data
        :return: a string
        """
        return '%s.%s' % (self.NAME, io_utils.DATA_EXT)

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        """Save data in base_save_dirpath
        Serializes variables specified in by self.__serializable_variables__()

        Save location:
            Data will be saved in the directory base_save_dirpath/scan.NAME_data/ (e.g. base_save_dirpath/dess_data/)

        Override:
            Override this method in the subscans to save additional information such as volumes, subvolumes,
                quantitative maps, etc.

            Call this function (super().save_data(base_save_dirpath)) before adding code to override this method

        :param base_save_dirpath: base directory to store data
        :param data_format: image data format (nifti, dicom, etc) that scan data should be stored as
        """

        # write other data as ref
        save_dirpath = self.__save_dir__(base_save_dirpath)
        filepath = os.path.join(save_dirpath, '%s.data' % self.NAME)

        metadata = dict()
        for variable_name in self.__serializable_variables__():
            metadata[variable_name] = self.__getattribute__(variable_name)

        io_utils.save_pik(filepath, metadata)

    def load_data(self, base_load_dirpath: str):
        """Load data in base_load_dirpath

        Save location:
            Data will be saved in the directory base_load_dirpath/scan.NAME/ (e.g. base_load_dirpath/dess_data/)

        Override:
            Override this method in the subscans to load additional information such as volumes, subvolumes,
                quantitative maps, etc.

            Call this function (super().load_data(base_save_dirpath)) before adding code to override this method

        :param base_load_dirpath: base directory to load data from
                                    should be the same as the base_save_dirpath in save_data

        :raise NotADirectoryError: if base_load_dirpath/scan.NAME_data/ does not exist
        """
        load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        if not os.path.isdir(load_dirpath):
            raise NotADirectoryError('%s does not exist' % load_dirpath)

        filepath = os.path.join(load_dirpath, '%s.data' % self.NAME)

        metadata = io_utils.load_pik(filepath)
        for key in metadata.keys():
            if hasattr(self, key):
                self.__setattr__(key, metadata[key])

        try:
            self.__load_dicom__()
        except:
            print('Dicom directory %s not found. Will try to load from %s' % (self.dicom_path, base_load_dirpath))

    def __save_dir__(self, dirpath: str, create_dir: bool = True):
        """Returns directory specific to this scan

        :param dirpath: base directory path to locate data directory for this scan
        :param create_dir: create the data directory
        :return: data directory for this scan
        """
        # folder_id = '%s-%03d' % (self.NAME, self.series_number)
        folder_id = self.NAME

        name_len = len(folder_id) + 2  # buffer
        if self.NAME in dirpath[-name_len:]:
            scan_dirpath = os.path.join(dirpath, folder_id)
        else:
            scan_dirpath = dirpath

        scan_dirpath = os.path.join(scan_dirpath, folder_id)

        if create_dir:
            scan_dirpath = io_utils.check_dir(scan_dirpath)

        return scan_dirpath

    def __serializable_variables__(self):
        return ['dicom_path', 'series_number']


class TargetSequence(ScanSequence):
    """Defines scans that have high enough resolution & SNR to be used as targets for segmentation"""

    @abstractmethod
    def segment(self, model: SegModel, tissue: Tissue) -> MedicalVolume:
        """
        Segment based on model
        :param model: a SegModel instance
        :param tissue: A Tissue instance
        :return: a MedicalVolume instance
        """
        pass


class NonTargetSequence(ScanSequence):
    """Defines scans that cannot serve as targets and have to be registered to some other target scan
    Examples: Cubequant, Cones
    """

    @abstractmethod
    def interregister(self, target_path: str, mask_path: str = None):
        """
        Register this scan to the target scan - save as parameter in scan (volumes, subvolumes, etc)

        If there are many subvolumes to interregister with the base scan, typically the following actions are used to
            reduce error accumulation with multiple registrations
            1. Pick the subvolume with the highest SNR. Call this the base moving image
            2. Register the base moving image to the target image using elastix
            3. Capture the transformation file(s) detailing the transforms to go from base moving image --> target image
            4. Apply these transformation(s) to the remaining subvolumes

        :param target_path: a filepath to a nifti file storing the target scan
                            This scan will serve as the base for registration
                            Note the best target scan will have a high SNR
        :param mask_path: path to mask to use to use as focus points for registration
        """
        pass

    def __split_volumes__(self, expected_num_subvolumes: int):
        """
        Split the volumes into multiple subvolumes based on the echo time
        Each subvolume represents a single volumes of slices acquired with the same TR and TE times

        For example, Cubequant uses 4 spin lock times -- this will produce 4 subvolumes

        :param expected_num_subvolumes: Expected number of subvolumes
        :return: dictionary of subvolumes mapping echo time index --> MedicalVolume, and a list of echo times in
                    ascending order
                    e.g. {0: MedicalVolume A, 1:MedicalVolume B}, [10, 50]
                            10 (at index 0 in the --> key 0 --> MedicalVolume A
                            50 --> key 1 --> MedicalVolume B
        """
        volumes = self.volumes

        assert len(volumes) == expected_num_subvolumes, "Expected %d subvolumes but got %d" % (expected_num_subvolumes,
                                                                                               len(volumes))
        num_echo_times = len(volumes)
        echo_times = []

        for i in range(num_echo_times):
            echo_time = float(volumes[i].headers[0].EchoTime)
            echo_times.append((i, echo_time))

        # Sort list of tuples (ind, echo_time) by echo_time
        ordered_echo_times = sorted(echo_times, key=lambda x: x[1])

        ordered_subvolumes_dict = dict()

        for i in range(num_echo_times):
            volume_ind, echo_time = ordered_echo_times[i]
            ordered_subvolumes_dict[i] = volumes[volume_ind]

        subvolumes_dict = ordered_subvolumes_dict

        echo_times = [x for _, x in echo_times]

        return subvolumes_dict, echo_times

    def __load_interregistered_files__(self, interregistered_dirpath: str):
        """Load the nifti files of the interregistered subvolumes
        These subvolumes have already been registered to some base scan using the interregister function

        :param interregistered_dirpath: path to interregistered directory folder (should be in the data folder specified
                                            for saving data)
        :return: dictionary of subvolumes mapping echo time index --> MedicalVolume

        :raise: ValueError:
                    1. Files are not of the name <INTEGER>.nii.gz (e.g. 0.nii.gz, 000.nii.gz, etc)
                    2. No interregistered files found in interregistered_dirpath
        """
        print('Loading interregistered files')
        if 'interregistered' not in interregistered_dirpath:
            raise ValueError('Invalid path for loading %s interregistered files' % self.NAME)

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError('No interregistered files found')

        indices = []
        subvolumes = []
        nifti_reader = NiftiReader()
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError('%s is not an interregisterd \'.gz.nii\' file.' % subfile)

            subfile_num = int(subfile_nums[0])
            indices.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume = nifti_reader.load(filepath)

            subvolumes.append(subvolume)

        assert len(indices) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError('No interregistered files found')

        subvolumes_dict = dict()
        for i in range(len(indices)):
            subvolumes_dict[indices[i]] = subvolumes[i]

        return subvolumes_dict

    def __dilate_mask__(self, mask_path: str, temp_path: str,
                        dil_rate: float = defaults.DEFAULT_MASK_DIL_RATE,
                        dil_threshold: float = defaults.DEFAULT_MASK_DIL_THRESHOLD):
        """Dilate mask using gaussian blur and write to disk to use with elastix

        :param mask_path: path to mask to use to use as focus points for registration, mask must be binary
        :param temp_path: path to store temporary data
        :param dil_rate: dilation rate (sigma)
        :param dil_threshold: threshold to binarize dilated mask - float between [0, 1]
        :return: the path to the dilated mask

        :raise FileNotFoundError:
                    1. filepath specified by mask_path is not found
        :raise ValueError:
                    1. dil_threshold not in range [0, 1]
        """

        if not os.path.isfile(mask_path):
            raise FileNotFoundError('File %s not found' % mask_path)

        if dil_threshold < 0 or dil_threshold > 1:
            raise ValueError('dil_threshold must be in range [0, 1]')

        mask = fio_utils.generic_load(mask_path, expected_num_volumes=1)

        dilated_mask = sni.gaussian_filter(np.asarray(mask.volume, dtype=np.float32),
                                           sigma=dil_rate) > dil_threshold
        fixed_mask = np.asarray(dilated_mask,
                                dtype=np.int8)
        fixed_mask_filepath = os.path.join(io_utils.check_dir(temp_path), 'dilated-mask.nii.gz')

        dilated_mask_volume = MedicalVolume(fixed_mask,
                                            affine=mask.affine)
        dilated_mask_volume.save_volume(fixed_mask_filepath)

        return fixed_mask_filepath

    def __interregister_base_file__(self, base_image_info: tuple, target_path: str, temp_path: str,
                                    mask_path: str = None,
                                    parameter_files=(fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE)):
        """Interregister the base moving image to the target image

        :param base_image_info: tuple of filepath, echo index (eg. 'scans/000.nii.gz, 0)
        :param target_path: filepath to target scan - should be in nifti (.nii.gz) format
        :param temp_path: path to store temporary data
        :param mask_path: path to mask to use to use as focus points for registration, mask must be binary
                            recommend that this is the path to a dilated version of the mask for registration purposes
        :param parameter_files: list of filepaths to elastix parameter files to use for transformations
        :return: tuple of the path to the transformed moving image and a list of filepaths to elastix transformations
                    (e.g. '/result.nii.gz', ['/tranformation0.txt', '/transformation1.txt'])
        """
        base_image_path, base_time_id = base_image_info
        # Register base image to the target image
        print('Registering %s (base image)' % base_image_path)
        transformation_files = []

        use_mask_arr = [False, True]
        reg_output = None
        moving_image = base_image_path

        for i in range(len(parameter_files)):
            use_mask = use_mask_arr[i]
            pfile = parameter_files[i]

            reg = Registration()
            reg.inputs.fixed_image = target_path
            reg.inputs.moving_image = moving_image
            reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_path,
                                                                     '%03d_param%i' % (base_time_id, i)))
            reg.inputs.parameters = pfile

            if use_mask and mask_path is not None:
                fixed_mask_filepath = self.__dilate_mask__(mask_path, temp_path)
                reg.inputs.fixed_mask = fixed_mask_filepath

            reg.terminal_output = fc.NIPYPE_LOGGING

            reg_output = reg.run()
            reg_output = reg_output.outputs
            assert reg_output is not None

            # update moving image to output
            moving_image = reg_output.warped_file

            transformation_files.append(reg_output.transform[0])

        return reg_output.warped_file, transformation_files

    def __apply_transform__(self, image_info, transformation_files, temp_path):
        """Apply transform(s) to moving image using transformix
        :param image_info: tuple of filepath, echo index (eg. 'scans/000.nii.gz, 0)
        :param transformation_files: list of filepaths to elastix transformations
        :param temp_path: path to store temporary data
        :return: filepath to warped file in nifti (.nii.gz) format
        """
        filename, image_id = image_info
        print('Applying transform %s' % filename)
        warped_file = ''
        for f in transformation_files:
            reg = ApplyWarp()
            reg.inputs.moving_image = filename if len(warped_file) == 0 else warped_file

            reg.inputs.transform_file = f
            reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_path,
                                                                     '%03d' % image_id))
            reg.terminal_output = fc.NIPYPE_LOGGING
            reg_output = reg.run()

            warped_file = reg_output.outputs.warped_file
            assert warped_file != ''

        return warped_file
