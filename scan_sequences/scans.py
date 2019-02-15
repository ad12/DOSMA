import os
import re
import warnings
from abc import ABC, abstractmethod
from time import gmtime, strftime

import numpy as np
import scipy.ndimage as sni
from natsort import natsorted
from nipype.interfaces.elastix import Registration, ApplyWarp

import defaults
import file_constants as fc
from data_io.med_volume import MedicalVolume
from utils import dicom_utils
from utils import io_utils


class ScanSequence(ABC):
    NAME = ''

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        """
        :param dicom_path: a path to the folder containing all dicoms for this scan
        :param dicom_ext: extension of these dicom files
        :param load_path: base path were data is stored

        In practice, only either dicom_path or load_path should be specified
        If both are specified, the dicom_path is used, and all information in load_path is ignored
        """
        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (self.NAME,
                                                                          strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        self.tissues = []
        self.dicom_path = os.path.abspath(dicom_path) if dicom_path is not None else None
        self.dicom_ext = dicom_ext

        self.volume = None
        self.ref_dicom = None

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

    def __load_dicom__(self):
        """
        Store the dicoms to MedicalVolume and store a list of the dicom headers
        """
        dicom_path = self.dicom_path
        dicom_ext = self.dicom_ext

        self.volume, self.refs_dicom = dicom_utils.load_dicom(dicom_path, dicom_ext)

        self.ref_dicom = self.refs_dicom[0]

    def get_dimensions(self):
        """
        Get shape of volume
        :return: tuple of ints
        """
        return self.volume.volume.shape

    def __add_tissue__(self, new_tissue):
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

    def save_data(self, base_save_dirpath):
        """Save data in base_save_dirpath
        Serializes variables specified in by self.__serializable_variables__()

        Save location:
            Data will be saved in the directory base_save_dirpath/scan.NAME_data/ (e.g. base_save_dirpath/dess_data/)

        Override:
            Override this method in the subscans to save additional information such as volumes, subvolumes,
                quantitative maps, etc.

            Call this function (super().save_data(base_save_dirpath)) before adding code to override this method

        :param base_save_dirpath: base directory to store data
        """

        # write other data as ref
        save_dirpath = self.__save_dir__(base_save_dirpath)
        filepath = os.path.join(save_dirpath, '%s.data' % self.NAME)

        metadata = dict()
        for variable_name in self.__serializable_variables__():
            metadata[variable_name] = self.__getattribute__(variable_name)

        io_utils.save_pik(filepath, metadata)

    def load_data(self, base_load_dirpath):
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
            self.__setattr__(key, metadata[key])

        self.__load_dicom__()

    def __save_dir__(self, dirpath, create_dir=True):
        """Returns directory specific to this scan

        :param dirpath: base directory path to locate data directory for this scan
        :param create_dir: create the data directory
        :return: data directory for this scan type
        """
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
    """Defines scans that have high enough resolution+SNR to be used as targets segmentation"""

    def preprocess_volume(self, volume):
        """
        Preprocess segmentation volume

        Default: whitening the volume (X - mean(X)) / std(X)

        :param volume: 3D segmentation volume (numpy array)
        :return: 3D preprocessed numpy array
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
    """Defines scans that cannot serve as targets and have to be registered to some other target scan
    Examples: Cubequant, Cones
    """

    @abstractmethod
    def interregister(self, target_path, mask_path=None):
        """
        Register this scan to the target scan - save as parameter in scan (volume, subvolumes, etc)

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

    def __split_volumes__(self, expected_num_subvolumes):
        """
        Split the volume into multiple subvolumes based on the echo time
        Each subvolume represents a single volume of slices acquired with the same TR and TE times

        For example, Cubequant uses 4 spin lock times -- this will produce 4 subvolumes

        :param expected_num_subvolumes: Expected number of subvolumes
        :return: dictionary of subvolumes mapping echo time index --> MedicalVolume, and a list of echo times in
                    ascending order
                    e.g. {0: MedicalVolume A, 1:MedicalVolume B}, [10, 50]
                            10 (at index 0 in the --> key 0 --> MedicalVolume A
                            50 --> key 1 --> MedicalVolume B

        :raise ValueError:
                    1. If subvolume sizes are not equal - aka len(ref_dicoms) % num_echos != 0
                    2. If # slices in subvolume != len(ref_dicoms) % num_echos
        """
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

        assert num_echo_times == expected_num_subvolumes

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
            if len(subvolumes_dict[key]) != num_slices_per_subvolume:
                raise ValueError('subvolume \'%s\' (echo_time %s) has %d slices. Expected %d slices' % (
                    key, echo_times[key - 1], len(subvolumes_dict[key]), num_slices_per_subvolume))

        for key in subvolumes_dict.keys():
            sv = subvolumes_dict[key]
            sv = MedicalVolume(volume.volume[..., sv], volume.pixel_spacing)

            subvolumes_dict[key] = sv

        return subvolumes_dict, echo_times

    def __load_interregistered_files__(self, interregistered_dirpath):
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

        mask = io_utils.load_nifti(mask_path)

        dilated_mask = sni.gaussian_filter(np.asarray(mask.volume, dtype=np.float32),
                                           sigma=dil_rate) > dil_threshold
        fixed_mask = np.asarray(dilated_mask,
                                dtype=np.int8)
        fixed_mask_filepath = os.path.join(io_utils.check_dir(temp_path), 'dilated-mask.nii.gz')

        dilated_mask_volume = MedicalVolume(fixed_mask, mask.pixel_spacing)
        dilated_mask_volume.save_volume(fixed_mask_filepath)

        return fixed_mask_filepath

    def __interregister_base_file__(self, base_image_info, target_path, temp_path, mask_path=None,
                                    parameter_files=[fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]):
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
