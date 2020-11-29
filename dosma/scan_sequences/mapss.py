import os
from copy import deepcopy
from typing import List

import numpy as np
from nipype.interfaces.elastix import Registration

from dosma.scan_sequences.scans import TargetSequence

from dosma import file_constants as fc
from dosma.data_io import format_io_utils as fio_utils
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.med_volume import MedicalVolume
from dosma.data_io.nifti_io import NiftiReader
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel
from dosma.quant_vals import QuantitativeValueType
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils
from dosma import quant_vals as qv
from dosma.utils.cmd_line_utils import ActionWrapper
from dosma.utils.fits import MonoExponentialFit

import logging
from typing import List, Sequence


__EXPECTED_NUM_ECHO_TIMES__ = 7

__INITIAL_T1_RHO_VAL__ = 70.0
__T1_RHO_LOWER_BOUND__ = 0
__T1_RHO_UPPER_BOUND__ = 500

__INITIAL_T2_VAL__ = 30.0
__T2_LOWER_BOUND__ = 0
__T2_UPPER_BOUND__ = 100

__DECIMAL_PRECISION__ = 3

__all__ = ["Mapss"]


class Mapss(TargetSequence):
    NAME = "mapss"

    def __init__(self, dicom_path=None, load_path=None, **kwargs):
        self.echo_times = None
        self.raw_volumes = None

        super().__init__(dicom_path=dicom_path, load_path=load_path, **kwargs)

        if dicom_path is not None:
            self.__intraregister__(self.volumes)

    def __load_dicom__(self):
        super().__load_dicom__()
        self.echo_times = [float(x.headers[0].EchoTime) for x in self.volumes]

    def __validate_scan__(self):
        return len(self.volumes) == 7

    def segment(self, model: SegModel, tissue: Tissue):
        """Currently not implemented."""
        raise NotImplementedError("This method is currently not implemented. "
                                  "Automatic segmentation model is currently being trained")

    def __intraregister__(self, volumes: List[MedicalVolume]):
        """Intraregister volumes.

        Sets `self.volumes` to intraregistered volumes.

        Args:
            volumes (list[MedicalVolume]): Volumes to register.

        Raises:
            TypeError: If `volumes` is not `list[MedicalVolume]`.
        """
        if (not volumes) or (type(volumes) is not list) or (len(volumes) != __EXPECTED_NUM_ECHO_TIMES__):
            raise TypeError("`volumes` must be of type List[MedicalVolume]")

        num_echos = len(volumes)

        logging.info("")
        logging.info("==" * 40)
        logging.info("Intraregistering...")
        logging.info("==" * 40)

        # temporarily save subvolumes as nifti file
        raw_volumes_base_path = io_utils.mkdirs(os.path.join(self.temp_path, "raw"))

        # Use first subvolume as a basis for registration - save in nifti format to use with elastix/transformix
        volume_files = []
        for echo_index in range(num_echos):
            filepath = os.path.join(raw_volumes_base_path, "{:03d}.nii.gz".format(echo_index))
            volume_files.append(filepath)

            volumes[echo_index].save_volume(filepath, data_format=ImageDataFormat.nifti)

        target_echo_index = 0
        target_image_filepath = volume_files[target_echo_index]

        nr = NiftiReader()
        intraregistered_volumes = [deepcopy(volumes[target_echo_index])]
        for echo_index in range(1, num_echos):
            moving_image = volume_files[echo_index]

            reg = Registration()
            reg.inputs.fixed_image = target_image_filepath
            reg.inputs.moving_image = moving_image
            reg.inputs.output_path = io_utils.mkdirs(os.path.join(self.temp_path,
                                                                     "intraregistered",
                                                                     "{:03d}".format(echo_index)))
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            reg.terminal_output = fc.NIPYPE_LOGGING
            logging.info("Registering {} -> {}".format(str(echo_index), str(target_echo_index)))
            tmp = reg.run()

            warped_file = tmp.outputs.warped_file
            intrareg_vol = nr.load(warped_file)

            # copy affine from original volume, because nifti changes loading accuracy
            intrareg_vol = MedicalVolume(volume=intrareg_vol.volume,
                                         affine=volumes[echo_index].affine,
                                         headers=deepcopy(volumes[echo_index].headers))

            intraregistered_volumes.append(intrareg_vol)

        self.raw_volumes = deepcopy(volumes)
        self.volumes = intraregistered_volumes

    def generate_t1_rho_map(self, tissue: Tissue = None, mask_path: str = None, num_workers: int = 0):
        """Generate 3D T1-rho map and r-squared fit map using mono-exponential fit across subvolumes acquired at
            different echo times.

        Args:
            tissue (Tissue): Tissue to generate quantitative value for.
            mask_path (str): File path to mask of ROI to analyze
            num_workers (int, optional): Number of subprocesses to use for fitting.
                If `0`, will execute on the main thread.

        Returns:
            qv.T1Rho: T1-rho fit for tissue.
        """
        echo_inds = range(4)
        bounds = (__T1_RHO_LOWER_BOUND__, __T1_RHO_UPPER_BOUND__)
        tc0 = __INITIAL_T1_RHO_VAL__
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(qv.T1Rho, echo_inds, tissue, bounds, tc0, decimal_precision, mask_path, num_workers)

        return qv_map

    def generate_t2_map(self, tissue: Tissue = None, mask_path: str = None, num_workers: int = 0):
        """Generate 3D T2 map and r-squared fit map using mono-exponential fit across subvolumes acquired at different
            echo times.

        Args:
            tissue (Tissue): Tissue to generate quantitative value for.
            mask_path (str): File path to mask of ROI to analyze
            num_workers (int, optional): Number of subprocesses to use for fitting.
                If `0`, will execute on the main thread.

        Returns:
            qv.T2: T2 fit for tissue.
        """
        echo_inds = [0, 4, 5, 6]
        bounds = (__T2_LOWER_BOUND__, __T2_UPPER_BOUND__)
        tc0 = __INITIAL_T2_VAL__
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(qv.T2, echo_inds, tissue, bounds, tc0, decimal_precision, mask_path, num_workers)

        return qv_map

    def __fitting_helper(self, qv_type: QuantitativeValueType, echo_inds: Sequence[int], tissue: Tissue,
                         bounds, tc0, decimal_precision, mask_path, num_workers):
        echo_info = [(self.echo_times[i], self.volumes[i]) for i in echo_inds]

        # sort by echo time
        echo_info = sorted(echo_info, key=lambda x: x[0])

        xs = [et for et, _ in echo_info]
        ys = [vol for _, vol in echo_info]

        # only calculate for focused region if a mask is available, this speeds up computation
        mask = tissue.get_mask()
        if not mask and mask_path:
            mask = fio_utils.generic_load(mask_path, expected_num_volumes=1)
            if tuple(np.unique(mask.volume)) != (0, 1):
                raise ValueError('mask_filepath must reference binary segmentation volume')

        mef = MonoExponentialFit(
            xs, ys,
            mask=mask,
            bounds=bounds,
            tc0=tc0,
            decimal_precision=decimal_precision,
            num_workers=num_workers
        )
        qv_map, r2 = mef.fit()

        quant_val_map = qv_type(qv_map)
        quant_val_map.add_additional_volume('r2', r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format):
        """Save data to disk.

        Data will be saved in the directory '`base_save_dirpath`/mapss/'.

        Serializes variables specified in by self.__serializable_variables__().

        Args:
            base_save_dirpath (str): Directory path where all data is stored.
            data_format (ImageDataFormat): Format to save data.
        """
        super().save_data(base_save_dirpath, data_format=data_format)

        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # Write echos.
        for i in range(len(self.volumes)):
            nii_registration_filepath = os.path.join(base_save_dirpath, "echo{:d}.nii.gz".format(i + 1))
            filepath = fio_utils.convert_image_data_format(nii_registration_filepath, data_format)
            self.volumes[i].save_volume(filepath, data_format=data_format)

    def load_data(self, base_load_dirpath):
        """Load data from disk.

        Data will be loaded from the directory '`base_load_dirpath`/mapss'.

        Args:
           base_load_dirpath (str): Directory path where all data is stored.

        Raises:
           NotADirectoryError: if `base_load_dirpath`/mapss/ does not exist.
        """
        super().load_data(base_load_dirpath)

        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        self.volumes = []

        # Load subvolumes from nifti file.
        for i in range(__EXPECTED_NUM_ECHO_TIMES__):
            nii_registration_filepath = os.path.join(base_load_dirpath, "echo{:d}.nii.gz".format(i + 1))
            subvolume = NiftiReader().load(nii_registration_filepath)
            self.volumes.append(subvolume)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['echo_times'])
        return var_names

    @classmethod
    def cmd_line_actions(cls):
        """Provide command line information (such as name, help strings, etc) as list of dictionary."""

        generate_t1_rho_map_action = ActionWrapper(name=cls.generate_t1_rho_map.__name__,
                                                   aliases=['t1_rho'],
                                                   param_help={
                                                       'mask_path': 'mask filepath (.nii.gz) to reduce computational time for fitting. Not required if loading data (ie. `--l` flag) for tissue with mask.'},
                                                   alternative_param_names={'mask_path': ['mask', 'mp']},
                                                   help='generate T1-rho map using mono-exponential fitting')
        generate_t2_map_action = ActionWrapper(name=cls.generate_t2_map.__name__,
                                               aliases=['t2'],
                                               param_help={
                                                   'mask_path': 'mask filepath (.nii.gz) to reduce computational time for fitting. Not required if loading data (ie. `--l` flag) for tissue with mask.'},
                                               alternative_param_names={'mask_path': ['mask', 'mp']},
                                               help='generate T2 map using mono-exponential fitting')

        return [(cls.generate_t1_rho_map, generate_t1_rho_map_action),
                (cls.generate_t2_map, generate_t2_map_action)]
