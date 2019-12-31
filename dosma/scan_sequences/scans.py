"""Abstract classes interface implementation of scan.

This module defines the abstract interfaces that can be extended by concrete scan types.

Different scan types produce images with different qualities (different quantitative parameters, resolutions, etc.).
    These scan types have different actions (or processing) associated with them.

A scan can have multiple volumes if multiple phases or echo times are used to image in the scan.
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

from dosma import file_constants as fc
from dosma.data_io import format_io_utils as fio_utils
from dosma.data_io.dicom_io import DicomReader
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.med_volume import MedicalVolume
from dosma.data_io.nifti_io import NiftiReader
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils

import logging


class ScanSequence(ABC):
    """Abstract class for scan sequences and corresponding analysis.

    All scan sequence classes should inherit from this abstract classes.

    In practice, only either `dicom_path` or `load_path` should be specified.

    If both are specified, the dicom_path is used, and all information in `load_path` is ignored.

    Args:
        dicom_path (str): Folder path to DICOM files.
        load_path (str): Base path where data is stored.
        **kwargs: Arbitrary keyword arguments.

    Kwargs:

        split_by (`str` or `tuple`, optional): DICOM field tag name or tag number used to group dicoms. Default depends
            on scan sequence - typically EchoNumber. `dicom_path` must be specified.
        ignore_ext (`bool`, optional): Ignore extension when loading DICOM files. `dicom_path` must be specified.

    Raises:
        NotADirectoryError: If `dicom_path` is not a valid directory.
        ValueError: If dicoms do not correspond to the expected sequence.

    """
    NAME = ''
    __DEFAULT_SPLIT_BY__ = 'EchoNumbers'

    def __init__(self, dicom_path: str = None, load_path: str = None, **kwargs):
        self.split_by = self.__DEFAULT_SPLIT_BY__
        self.ignore_ext = False

        kwargs_str = ["split_by", "ignore_ext"]
        for k in kwargs_str:
            if k in kwargs:
                self.__setattr__(k, kwargs.get(k))

        self.temp_path = os.path.join(fc.TEMP_FOLDER_PATH, self.NAME, strftime("%Y-%m-%d-%H-%M-%S", localtime()))
        self.tissues = []
        self.dicom_path = os.path.abspath(dicom_path) if dicom_path is not None else None

        self.volumes = None
        self.ref_dicom = None
        self.series_number = None

        # check if dicom path exists
        if (dicom_path is not None) and (not os.path.isdir(dicom_path)):
            if load_path is not None:
                warnings.warn("Dicom_path {} not found. Will load data from {}".format(dicom_path, load_path))
            else:
                raise NotADirectoryError("{} is not a directory".format(dicom_path))

        # Only use dicoms if the path exists and path contains files ending in dicom_ext
        is_dicom_available = (dicom_path is not None) and (os.path.isdir(dicom_path))

        # Only load data if dicom path is not given or doesn't exist, else assume user wants to rewrite information
        if load_path and not is_dicom_available:
            self.load_data(load_path)

        if is_dicom_available:
            self.__load_dicom__()

        if not self.__validate_scan__():
            raise ValueError("dicoms in '{}' do not correspond to {} sequence".format(self.dicom_path, self.NAME))

    @abstractmethod
    def __validate_scan__(self) -> bool:
        """Validate this scan (usually done by checking dicom header tags, if available).

        Returns:
            bool: `True` if scan metadata is valid, `False` otherwise.
        """
        pass

    def __load_dicom__(self):
        """Load data from dicom path.
        """
        split_by = self.split_by

        dicom_path = self.dicom_path

        if dicom_path is None or not os.path.isdir(dicom_path):
            raise NotADirectoryError('%s not found' % dicom_path)

        dr = DicomReader()

        self.volumes = dr.load(dicom_path, group_by=split_by, ignore_ext=self.ignore_ext)

        self.ref_dicom = self.volumes[0].headers[0]

        self.__set_series_number__(self.ref_dicom.SeriesNumber)

    def __set_series_number__(self, sn: int):
        """Set series number.

        Args:
            sn (int): Series number.
        """
        if self.series_number is not None:
            assert self.series_number == sn, "Series numbers must be identical if loading the same scan"
            return
        else:
            self.series_number = sn

    def get_dimensions(self):
        """Get shape of volumes.

        All volumes in scan are assumed to be same dimension.

        Returns:
            tuple[int]: Shape of volumes in scan.
        """
        return self.volumes[0].volume.shape

    def __add_tissue__(self, new_tissue: Tissue):
        """Add a tissue to the list of tissues associated with this scan.

        Args:
            new_tissue (Tissue): Tissue to add.

        Raises:
            ValueError: If tissue already exists in list. For example, we cannot add FemoralCartilage twice to the list
                of tissues
        """
        contains_tissue = any([tissue.ID == new_tissue.ID for tissue in self.tissues])
        if contains_tissue:
            raise ValueError("Tissue already exists")

        self.tissues.append(new_tissue)

    def __data_filename__(self):
        """Get filename for storing serialized data.

        Returns:
            str: File name for pickled data.
        """
        return '%s.data' % self.NAME

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format):
        """Save data to disk.

        Data will be saved in the directory '`base_save_dirpath`/scan.NAME_data/'
            (e.g. '`base_save_dirpath`/dess_data/').

        Serializes variables specified in by self.__serializable_variables__().

        Override this method to save additional information such as volumes, subvolumes, quantitative maps, etc.
            In override, Call this function (super().save_data(base_save_dirpath)) before adding code to override this
            method.

        Args:
            base_save_dirpath (str): Directory path where all data is stored.
            data_format (ImageDataFormat): Format to save data.
        """

        # Write data as ref.
        save_dirpath = self.__save_dir__(base_save_dirpath)
        filepath = os.path.join(save_dirpath, '%s.data' % self.NAME)

        metadata = dict()
        for variable_name in self.__serializable_variables__():
            metadata[variable_name] = self.__getattribute__(variable_name)

        io_utils.save_pik(filepath, metadata)

    def load_data(self, base_load_dirpath: str):
        """Load data from disk.

        Data will be loaded from the directory '`base_load_dirpath`/`scan.NAME`' (e.g. '`base_load_dirpath'/dess/').

        Override this method to load additional information such as volumes, subvolumes, quantitative maps, etc.
            In override, Call this function (super().save_data(base_load_dirpath)) before adding code to override this
            method.

        Args:
            base_load_dirpath (str): Directory path where all data is stored.

        Raises:
            NotADirectoryError: if base_load_dirpath/scan.NAME/ does not exist.
        """
        load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        if not os.path.isdir(load_dirpath):
            raise NotADirectoryError("{} does not exist".format(load_dirpath))

        file_path = os.path.join(load_dirpath, "{}.data".format(self.NAME))

        metadata = io_utils.load_pik(file_path)
        for key in metadata.keys():
            if hasattr(self, key):
                self.__setattr__(key, metadata[key])

        try:
            self.__load_dicom__()
        except:
            logging.info("Dicom directory {} not found. Will try to load from {}".format(self.dicom_path, base_load_dirpath))

    def __save_dir__(self, dir_path: str, create_dir: bool = True):
        """Returns directory path specific to this scan.

        Formatted as '`base_load_dirpath`/`scan.NAME`'.

        Args:
            dir_path (str): Directory path where all data is stored.
            create_dir (`bool`, optional): If `True`, creates directory if it doesn't exist.

        Returns:
            str: Data directory path for this scan.
        """
        # folder_id = '%s-%03d' % (self.NAME, self.series_number)
        folder_id = self.NAME

        name_len = len(folder_id) + 2  # buffer
        if self.NAME in dir_path[-name_len:]:
            scan_dirpath = os.path.join(dir_path, folder_id)
        else:
            scan_dirpath = dir_path

        scan_dirpath = os.path.join(scan_dirpath, folder_id)

        if create_dir:
            scan_dirpath = io_utils.mkdirs(scan_dirpath)

        return scan_dirpath

    def __serializable_variables__(self):
        """Variables to serialize.

        Not all variables can be serialized.

        To add scan specific variables, override this method.
        """
        return ['dicom_path', 'series_number', 'split_by', 'ignore_ext']


class TargetSequence(ScanSequence):
    """Abstract class for scans that support segmentation of tissues.

    Scans that have high resolution & signal-to-noise ratio (SNR) are typically used for segmentation. Because of this
        property, they are often referred to as "target scans". This allows them to serve as a good template that other
        lower SNR/resolution scans can be registered to.
    """

    @abstractmethod
    def segment(self, model: SegModel, tissue: Tissue) -> MedicalVolume:
        """Segment tissue in scan.

        Args:
            model (SegModel): Model to use for segmenting scans.
            tissue (Tissue): The tissue to segment.

        Returns:
            MedicalVolume: Binary mask for segmented region.
        """
        pass


class NonTargetSequence(ScanSequence):
    """Abstract class for scans that cannot serve as targets and have to be registered to some other target scan.

    Examples: Cubequant, Cones
    """

    @abstractmethod
    def interregister(self, target_path: str, mask_path: str = None):
        """Register this scan to the target scan - save as parameter in scan (volumes, subvolumes, etc).

        We use the term "interregister" to refer to registration between volumes of different scans. Conversely,
            "intraregister" refers to registering volumes from the same scan.

        If there are many subvolumes to interregister with the base scan, typically the following actions are used to
            reduce error accumulation with multiple registrations.
            1. Pick the volume with the highest SNR. Call this the base moving image.
            2. Register the base moving image to the target image using elastix.
            3. Capture the transformation file(s) detailing the transforms to go from base moving image -> target image.
            4. Apply these transformation(s) to the remaining volumes.

        Args:
            target_path (str): Path to NIfTI file storing scan. This scan will serve as the base for registration. Note
                the best target scan will have a high SNR.

            mask_path (str): Path to mask to use to use as focus points for registration.
        """
        pass

    def __split_volumes__(self, expected_num_subvolumes: int):
        """Split the scan into multiple volumes based on the echo time.

        Each volume represents a volumes of slices acquired with the same TR and TE times. For example, Cubequant uses 4
            spin lock times -- this will produce 4 volumes.

        Args:
            expected_num_subvolumes (int): Expected number of volumes that should be in the scan.

        Returns:
        `dict[int, MedicalVolume]`, `list[float]`: A dictionary mapping echo time index -> MedicalVolume and a list of
            echo times in ascending order.

            e.g.: {0: MedicalVolume A, 1:MedicalVolume B}, [10, 50]

            10 (at index 0 in the --> key 0 --> MedicalVolume A, 50 --> key 1 --> MedicalVolume B)
        """
        volumes = self.volumes

        if len(volumes) != expected_num_subvolumes:
            raise ValueError("Expected %d subvolumes but got %d" % (expected_num_subvolumes, len(volumes)))

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
        """Load the NIfTI files of the interregistered subvolumes.

        These subvolumes have already been registered to some target scan using the `interregister` function.

        Args:
            interregistered_dirpath (str): Directory path where interregistered volumes are stored.

        Returns:
        `dict[int, MedicalVolume]`, `list[float]`: A dictionary mapping echo time index -> MedicalVolume

        Raises:
            ValueError: If files are not of the name `<INTEGER>.nii.gz` (e.g. `0.nii.gz`, `000.nii.gz`, etc.)
                or if no interregistered files found in interregistered_dirpath.
        """
        logging.info("Loading interregistered files")
        if "interregistered" not in interregistered_dirpath:
            raise ValueError("Invalid path for loading {} interregistered files".format(self.NAME))

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError("No interregistered files found")

        indices = []
        subvolumes = []
        nifti_reader = NiftiReader()
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError("{} is not an interregistered '.gz.nii' file.".format(subfile))

            subfile_num = int(subfile_nums[0])
            indices.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume = nifti_reader.load(filepath)

            subvolumes.append(subvolume)

        assert len(indices) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError("No interregistered files found")

        subvolumes_dict = dict()
        for i in range(len(indices)):
            subvolumes_dict[indices[i]] = subvolumes[i]

        return subvolumes_dict

    def __dilate_mask__(self, mask_path: str, temp_path: str,
                        dil_rate: float = preferences.mask_dilation_rate,
                        dil_threshold: float = preferences.mask_dilation_threshold):
        """Dilate mask using gaussian blur and write to disk to use with Elastix.

        Args:
            mask_path (str): File path for mask to use to use as focus points for registration. Mask must be binary.
            temp_path (str): Directory path to store temporary data.
            dil_rate (`float`, optional): Dilation rate (sigma). Defaults to `preferences.mask_dilation_rate`.
            dil_threshold (`float`, optional): Threshold to binarize dilated mask. Must be between [0, 1]. Defaults to
                `preferences.mask_dilation_threshold`.

        Returns:
            str: File path of dilated mask.

        Raises:
            FileNotFoundError: If `mask_path` not valid file.
            ValueError: If `dil_threshold` not in range [0, 1].
        """

        if not os.path.isfile(mask_path):
            raise FileNotFoundError("File {} not found".format(mask_path))

        if dil_threshold < 0 or dil_threshold > 1:
            raise ValueError("'dil_threshold' must be in range [0, 1]")

        mask = fio_utils.generic_load(mask_path, expected_num_volumes=1)

        dilated_mask = sni.gaussian_filter(np.asarray(mask.volume, dtype=np.float32),
                                           sigma=dil_rate) > dil_threshold
        fixed_mask = np.asarray(dilated_mask,
                                dtype=np.int8)
        fixed_mask_filepath = os.path.join(io_utils.mkdirs(temp_path), "dilated-mask.nii.gz")

        dilated_mask_volume = MedicalVolume(fixed_mask,
                                            affine=mask.affine)
        dilated_mask_volume.save_volume(fixed_mask_filepath)

        return fixed_mask_filepath

    def __interregister_base_file__(self, base_image_info: tuple, target_path: str, temp_path: str,
                                    mask_path: str = None,
                                    parameter_files=(fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE)):
        """Interregister the base moving image to the target image.

        Args:
            base_image_info (tuple[str, int]): File path, echo index (eg. 'scans/000.nii.gz, 0).
            target_path (str): File path to target scan. Must be in nifti (.nii.gz) format.
            temp_path (str): Directory path to store temporary data.
            mask_path (str): Path to mask to use to use as focus points for registration. Mask must be binary. Recommend
                using dilated mask.
            parameter_files (list[str]): Transformix parameter files to use for transformations.

        Returns:
            tuple[str, list[str]): File path to the transformed moving image and a list of file paths to elastix
                transformations (e.g. '/result.nii.gz', ['/tranformation0.txt', '/transformation1.txt']).
        """
        base_image_path, base_time_id = base_image_info

        # Register base image to the target image.
        logging.info("Registering %s (base image)".format(base_image_path))
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
            reg.inputs.output_path = io_utils.mkdirs(os.path.join(temp_path,
                                                                     '{:03d}_param{}'.format(base_time_id, i)))
            reg.inputs.parameters = pfile

            if use_mask and mask_path is not None:
                fixed_mask_filepath = self.__dilate_mask__(mask_path, temp_path)
                reg.inputs.fixed_mask = fixed_mask_filepath

            reg.terminal_output = fc.NIPYPE_LOGGING

            reg_output = reg.run()
            reg_output = reg_output.outputs
            assert reg_output is not None

            # Update moving image to output.
            moving_image = reg_output.warped_file

            transformation_files.append(reg_output.transform[0])

        return reg_output.warped_file, transformation_files

    def __apply_transform__(self, image_info, transformation_files, temp_path):
        """Apply transform(s) to moving image using Transformix.

        Args:
            image_info (tuple[str, int]): File path, echo index (eg. 'scans/000.nii.gz, 0).
            transformation_files (list[str]): Ordered collection of paths to elastix transformation files.
            temp_path (str): Directory path to store temporary data.

        Returns:
            str: File path to warped file in NIfTI format.
        """
        filename, image_id = image_info
        logging.info("Applying transform {}".format(filename))
        warped_file = ''
        for f in transformation_files:
            reg = ApplyWarp()
            reg.inputs.moving_image = filename if len(warped_file) == 0 else warped_file

            reg.inputs.transform_file = f
            reg.inputs.output_path = io_utils.mkdirs(os.path.join(temp_path,
                                                                     "{:03d}".format(image_id)))
            reg.terminal_output = fc.NIPYPE_LOGGING
            reg_output = reg.run()

            warped_file = reg_output.outputs.warped_file
            assert warped_file != ""

        return warped_file
