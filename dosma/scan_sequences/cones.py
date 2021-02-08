"""Ultra-short Echo Time Cones (UTE-Cones)."""
import logging
import os

import numpy as np
from natsort import natsorted

from dosma import file_constants as fc
from dosma import quant_vals as qv
from dosma.data_io import NiftiReader
from dosma.data_io import format_io_utils as fio_utils
from dosma.data_io.med_volume import MedicalVolume
from dosma.scan_sequences.scans import NonTargetSequence
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils
from dosma.utils.cmd_line_utils import ActionWrapper
from dosma.utils.fits import MonoExponentialFit

__all__ = ["Cones"]

__EXPECTED_NUM_ECHO_TIMES__ = 4

__INITIAL_T2_STAR_VAL__ = 30.0

__T2_STAR_LOWER_BOUND__ = 0
__T2_STAR_UPPER_BOUND__ = np.inf
__T2_STAR_DECIMAL_PRECISION__ = 3


class Cones(NonTargetSequence):
    """UTE-Cones MRI sequence.

    Ultra-short echo time cones (UTE-Cones) is a :math:`T_2^*`-weighted sequence.
    In practice, many of these scans are low resolution and are ofter interregistered
    with higher-resolution scans. This can be done with :meth:`Cones.interregister`.

    References:
        Qian Y, Williams AA, Chu CR, Boada FE. Multicomponent T2* mapping of
        knee cartilage: technical feasibility ex vivo.
        Magnetic resonance in medicine 2010;64(5):1426-1431."
    """

    NAME = "cones"

    def __init__(self, volumes):
        super().__init__(volumes)
        self.subvolumes = None
        self.echo_times = None
        self.intraregistered_data = None

        if self.ref_dicom is not None:
            self.subvolumes, self.echo_times = self.__split_volumes__(__EXPECTED_NUM_ECHO_TIMES__)

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
        base_echo_time, base_image = (
            len(echo_time_inds) - 1,
            raw_filepaths[len(echo_time_inds) - 1],
        )

        temp_interregistered_dirpath = io_utils.mkdirs(
            os.path.join(self.temp_path, "interregistered")
        )

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
            parameter_files = [
                fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE,
                fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE,
            ]

        warped_file, transformation_files = self.__interregister_base_file__(
            (base_image, base_echo_time),
            target_path,
            temp_interregistered_dirpath,
            mask_path=target_mask_path,
            parameter_files=parameter_files,
        )
        warped_files = [(base_echo_time, warped_file)]

        # Load the transformation file. Apply same transform to the remaining images
        for echo_time, filename in files_to_warp:
            warped_file = self.__apply_transform__(
                (filename, echo_time), transformation_files, temp_interregistered_dirpath
            )
            # append the last warped file - this has all the transforms applied
            warped_files.append((echo_time, warped_file))

        # copy each of the interregistered warped files to their own output
        nifti_reader = NiftiReader()
        subvolumes = dict()
        for echo_time, warped_file in warped_files:
            subvolumes[echo_time] = nifti_reader.load(warped_file)

        self.subvolumes = subvolumes

    def generate_t2_star_map(self, tissue: Tissue, mask_path: str = None, num_workers: int = 0):
        """
        Generate 3D :math:`T_2^* map and r-squared fit map using mono-exponential fit
        across subvolumes acquired at different echo times.

        :math:`T_2^* map is also added to the tissue.

        Args:
            tissue (Tissue): Tissue to generate quantitative value for.
            mask_path (:obj:`str`, optional): File path to mask of ROI to analyze.
                If specified, only voxels specified by mask will be fit.
                This can considerably speed up computation.
            num_workers (int, optional): Number of subprocesses to use for fitting.
                If `0`, will execute on the main thread.

        Returns:
            qv.T2Star: :math:`T_2^* fit for tissue.

        Raises:
            ValueError: If ``mask_path`` corresponds to non-binary volume.
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
            spin_lock_times,
            subvolumes_list,
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

    def _save(self, metadata, save_dir, fname_fmt=None, **kwargs):
        default_fmt = {
            "subvolumes": os.path.abspath(os.path.join(save_dir, "interregistered/{}")),
            MedicalVolume: "echo-{}",
        }
        default_fmt.update(fname_fmt if fname_fmt else {})
        return super()._save(metadata, save_dir, fname_fmt=default_fmt, **kwargs)

    @classmethod
    def from_dict(cls, data, force: bool):
        interregistered_dirpath = os.path.dirpath(data.pop("subvolumes")[0])
        scan = super().from_dict(data, force=force)
        scan.__load_interregistered_files__(interregistered_dirpath)

    @classmethod
    def cmd_line_actions(cls):
        """
        Provide command line information (such as name, help strings, etc)
        as list of dictionary.
        """

        interregister_action = ActionWrapper(
            name=cls.interregister.__name__,
            help="register to another scan",
            param_help={
                "target_path": "path to target image in nifti format (.nii.gz)",
                "target_mask_path": "path to target mask in nifti format (.nii.gz)",
            },
            alternative_param_names={
                "target_path": ["tp", "target"],
                "target_mask_path": ["tm", "target_mask"],
            },
        )
        generate_t2star_map_action = ActionWrapper(
            name=cls.generate_t2_star_map.__name__,
            help="generate T2-star map",
            param_help={
                "mask_path": "Mask used for fitting select voxels - " "in nifti format (.nii.gz)"
            },
            aliases=["t2_star"],
        )

        return [
            (cls.interregister, interregister_action),
            (cls.generate_t2_star_map, generate_t2star_map_action),
        ]
