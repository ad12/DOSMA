"""Ultra-short Echo Time Cones (UTE-Cones).

Paper:
    Qian Y, Williams AA, Chu CR, Boada FE. "Multicomponent T2* mapping of knee cartilage: technical feasibility ex vivo."
    Magnetic resonance in medicine 2010;64(5):1426-1431."
"""
import os

import numpy as np
from natsort import natsorted

from dosma.scan_sequences.scans import NonTargetSequence

from dosma import file_constants as fc, quant_vals as qv
from dosma.data_io import ImageDataFormat, NiftiReader
from dosma.data_io import format_io_utils as fio_utils
from dosma.defaults import preferences
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils
from dosma.utils.cmd_line_utils import ActionWrapper
from dosma.utils.fits import MonoExponentialFit

import logging

__all__ = ["Cones"]

__EXPECTED_NUM_ECHO_TIMES__ = 4

__INITIAL_T2_STAR_VAL__ = 30.0

__T2_STAR_LOWER_BOUND__ = 0
__T2_STAR_UPPER_BOUND__ = np.inf
__T2_STAR_DECIMAL_PRECISION__ = 3


class Cones(NonTargetSequence):
    """Handles analysis for Cones scan sequence."""
    NAME = "cones"

    def __init__(self, dicom_path=None, load_path=None, **kwargs):
        self.subvolumes = None
        self.echo_times = None
        super().__init__(dicom_path=dicom_path, load_path=load_path, **kwargs)

        if dicom_path is not None:
            self.subvolumes, self.echo_times = self.__split_volumes__(__EXPECTED_NUM_ECHO_TIMES__)

        if self.subvolumes is None:
            raise ValueError("Either dicom_path or load_path must be specified")

    def __validate_scan__(self):
        return True

    def __load_dicom__(self):
        super().__load_dicom__()

        echo_times = []
        for vol in self.volumes:
            echo_times.append(float(vol.headers[0].EchoTime))

        self.echo_times = natsorted(echo_times)

    def interregister(self, target_path: str, target_mask_path: str = None):
        temp_raw_dirpath = io_utils.mkdirs(os.path.join(self.temp_path, "raw"))
        subvolumes = self.subvolumes

        raw_filepaths = dict()

        echo_time_inds = natsorted(list(subvolumes.keys()))

        for i in range(len(echo_time_inds)):
            raw_filepath = os.path.join(temp_raw_dirpath, "{:03d}.nii.gz".format(i))
            subvolumes[i].save_volume(raw_filepath)
            raw_filepaths[i] = raw_filepath

        # last echo should be base
        base_echo_time, base_image = len(echo_time_inds) - 1, raw_filepaths[len(echo_time_inds) - 1]

        temp_interregistered_dirpath = io_utils.mkdirs(os.path.join(self.temp_path, "interregistered"))

        logging.info("")
        logging.info("==" * 40)
        logging.info("Interregistering...")
        logging.info("Target: {}".format(target_path))
        if target_mask_path is not None:
            logging.info("Mask: {}".format(target_mask_path))
        logging.info("==" * 40)

        files_to_warp = []
        for echo_time_ind in raw_filepaths.keys():
            if echo_time_ind == base_echo_time:
                continue
            filepath = raw_filepaths[echo_time_ind]
            files_to_warp.append((echo_time_ind, filepath))

        if not target_mask_path:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
        else:
            parameter_files = [fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE]

        warped_file, transformation_files = self.__interregister_base_file__((base_image, base_echo_time),
                                                                             target_path,
                                                                             temp_interregistered_dirpath,
                                                                             mask_path=target_mask_path,
                                                                             parameter_files=parameter_files)
        warped_files = [(base_echo_time, warped_file)]

        # Load the transformation file. Apply same transform to the remaining images
        for echo_time, filename in files_to_warp:
            warped_file = self.__apply_transform__((filename, echo_time),
                                                   transformation_files,
                                                   temp_interregistered_dirpath)
            # append the last warped file - this has all the transforms applied
            warped_files.append((echo_time, warped_file))

        # copy each of the interregistered warped files to their own output
        nifti_reader = NiftiReader()
        subvolumes = dict()
        for echo_time, warped_file in warped_files:
            subvolumes[echo_time] = nifti_reader.load(warped_file)

        self.subvolumes = subvolumes

    def generate_t2_star_map(self, tissue: Tissue, mask_path: str = None, num_workers: int = 0):
        """Generate 3D T2-star map and r-squared fit map using mono-exponential fit across subvolumes acquired at
            different echo times.

        T2-star map is also added to the tissue.

        Args:
            tissue (Tissue): Tissue to generate quantitative value for.
            mask_path (:obj:`str`, optional): File path to mask of ROI to analyze. If specified, only voxels specified
                by mask will be fit. Speeds up computation. Defaults to `None`.
            num_workers (int, optional): Number of subprocesses to use for fitting.
                If `0`, will execute on the main thread.

        Returns:
            qv.T2Star: T2-star fit for tissue.

        Raises:
            ValueError: If `mask_path` specifies volume with values other than `0` or `1` (i.e. not binary).
        """
        # only calculate for focused region if a mask is available, this speeds up computation
        mask = tissue.get_mask()
        if (not mask or np.sum(mask.volume) == 0) and mask_path:
            mask = fio_utils.generic_load(mask_path, expected_num_volumes=1)
            if tuple(np.unique(mask.volume)) != (0, 1):
                raise ValueError("`mask_path` must reference binary segmentation volume")

        spin_lock_times = []
        subvolumes_list = []

        for echo_time in self.subvolumes.keys():
            spin_lock_times.append(echo_time)
            subvolumes_list.append(self.subvolumes[echo_time])

        mef = MonoExponentialFit(
            spin_lock_times, subvolumes_list,
            mask=mask,
            bounds=(__T2_STAR_LOWER_BOUND__, __T2_STAR_UPPER_BOUND__),
            tc0=__INITIAL_T2_STAR_VAL__,
            decimal_precision=__T2_STAR_DECIMAL_PRECISION__,
            num_workers=num_workers,
        )

        t2star_map, r2 = mef.fit()

        quant_val_map = qv.T2Star(t2star_map)
        quant_val_map.add_additional_volume("r2", r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format):
        """Save data to disk.

        Data will be saved in the directory '`base_save_dirpath`/cones/'.

        Serializes variables specified in by self.__serializable_variables__().

        Args:
            base_save_dirpath (str): Directory path where all data is stored.
            data_format (ImageDataFormat): Format to save data.
        """
        super().save_data(base_save_dirpath, data_format=data_format)
        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # Save interregistered files
        interregistered_dirpath = os.path.join(base_save_dirpath, 'interregistered')

        for spin_lock_time in self.subvolumes.keys():
            filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % spin_lock_time)
            self.subvolumes[spin_lock_time].save_volume(filepath)

    def load_data(self, base_load_dirpath: str):
        """Load data from disk.

        Data will be loaded from the directory '`base_load_dirpath`/cones'.

        Args:
           base_load_dirpath (str): Directory path where all data is stored.

        Raises:
           NotADirectoryError: if `base_load_dirpath`/cones/ does not exist.
        """
        super().load_data(base_load_dirpath)
        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(base_load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['echo_times'])

        return var_names

    @classmethod
    def cmd_line_actions(cls):
        """Provide command line information (such as name, help strings, etc) as list of dictionary."""

        interregister_action = ActionWrapper(name=cls.interregister.__name__,
                                             help='register to another scan',
                                             param_help={
                                                 'target_path': 'path to target image in nifti format (.nii.gz)',
                                                 'target_mask_path': 'path to target mask in nifti format (.nii.gz)'},
                                             alternative_param_names={'target_path': ['tp', 'target'],
                                                                      'target_mask_path': ['tm', 'target_mask']})
        generate_t2star_map_action = ActionWrapper(name=cls.generate_t2_star_map.__name__,
                                                   help='generate T2-star map',
                                                   param_help={
                                                       'mask_path': 'Mask used for fitting select voxels - '
                                                                    'in nifti format (.nii.gz)',
                                                   },
                                                   aliases=['t2_star'])

        return [(cls.interregister, interregister_action), (cls.generate_t2_star_map, generate_t2star_map_action)]
