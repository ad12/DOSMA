"""Ultra-short Echo Time Cones (UTE-Cones)."""
import logging
import os
from typing import Sequence

import numpy as np

from dosma import file_constants as fc
from dosma.core import quant_vals as qv
from dosma.core.fitting import MonoExponentialFit
from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.nifti_io import NiftiReader
from dosma.core.med_volume import MedicalVolume
from dosma.core.registration import apply_warp, register
from dosma.scan_sequences.scans import NonTargetSequence
from dosma.tissues.tissue import Tissue
from dosma.utils.cmd_line_utils import ActionWrapper

__all__ = ["Cones"]

__EXPECTED_NUM_ECHO_TIMES__ = 4

__INITIAL_T2_STAR_VAL__ = 30.0

__T2_STAR_LOWER_BOUND__ = 0
__T2_STAR_UPPER_BOUND__ = np.inf
__T2_STAR_DECIMAL_PRECISION__ = 3

_logger = logging.getLogger(__name__)


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

    def __init__(self, volumes, echo_times: Sequence[float] = None):
        super().__init__(volumes)

        if echo_times is None:
            try:
                if all(x.headers() is not None for x in self.volumes):
                    echo_times = [x.get_metadata("EchoTime", float) for x in self.volumes]
            except (KeyError, AttributeError, RuntimeError) as e:
                raise ValueError(
                    f"Could not extract echo times from header. "
                    f"Please specify `echo_times` argument - {e}"
                )

        self.echo_times = echo_times

    def interregister(self, target_path: str, target_mask_path: str = None):
        volumes = self.volumes
        echo_times = self.echo_times
        idxs = np.argsort(echo_times)

        echo_times = [echo_times[i] for i in idxs]
        volumes = [volumes[i] for i in idxs]
        nr = NiftiReader()
        out_path = os.path.join(self.temp_path, "interregistered")
        os.makedirs(out_path, exist_ok=True)

        # TODO: Make these into parameters
        num_threads = 2
        num_workers = 0
        verbose = True

        if verbose:  # pragma: no cover
            _logger.info("")
            _logger.info("==" * 40)
            _logger.info("Interregistering...")
            _logger.info("Target: {}".format(target_path))
            if target_mask_path is not None:
                _logger.info("Mask: {}".format(target_mask_path))
            _logger.info("==" * 40)

        # Target mask path has to be dilated.
        if target_mask_path:
            target_mask_path = self.__dilate_mask__(target_mask_path, out_path)
            parameter_files = [
                fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE,
                fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE,
            ]
            use_mask = [False, True]
        else:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
            use_mask = None

        # Last echo should be the base.
        base, moving = volumes[-1], volumes[:-1]

        out_reg, _ = register(
            target_path,
            base,
            parameters=parameter_files,
            output_path=out_path,
            sequential=True,
            collate=True,
            num_workers=num_workers,
            num_threads=num_threads,
            return_volumes=False,
            target_mask=target_mask_path,
            use_mask=use_mask,
            rtype=tuple,
            show_pbar=verbose,
        )
        out_reg = out_reg[0]

        reg_vols = []
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))
        reg_vols.append(nr.load(out_reg.warped_file))  # base volume is last

        # Undo sorting by echo time.
        reverse_idxs = {v: i for i, v in enumerate(idxs)}
        reg_vols = [reg_vols[reverse_idxs[k]] for k in sorted(reverse_idxs.keys())]

        self.volumes = reg_vols

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
        if mask_path is not None:
            mask = (
                fio_utils.generic_load(mask_path, expected_num_volumes=1)
                if isinstance(mask_path, (str, os.PathLike))
                else mask_path
            )

        spin_lock_times = self.echo_times
        subvolumes_list = self.volumes

        mef = MonoExponentialFit(
            bounds=(__T2_STAR_LOWER_BOUND__, __T2_STAR_UPPER_BOUND__),
            tc0="polyfit",
            decimal_precision=__T2_STAR_DECIMAL_PRECISION__,
            num_workers=num_workers,
            verbose=True,
        )

        t2star_map, r2 = mef.fit(spin_lock_times, subvolumes_list, mask=mask)

        quant_val_map = qv.T2Star(t2star_map)
        quant_val_map.add_additional_volume("r2", r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def _save(self, metadata, save_dir, fname_fmt=None, **kwargs):
        default_fmt = {MedicalVolume: "echo-{}"}
        default_fmt.update(fname_fmt if fname_fmt else {})
        return super()._save(metadata, save_dir, fname_fmt=default_fmt, **kwargs)

    @classmethod
    def from_dict(cls, data, force: bool = False) -> "Cones":
        interregistered_dirpath = None
        if "subvolumes" in data:
            interregistered_dirpath = os.path.dirname(data.pop("subvolumes")[0])
        scan: Cones = super().from_dict(data, force=force)
        if interregistered_dirpath is not None:
            subvolumes = scan.__load_interregistered_files__(interregistered_dirpath)
            cls.volumes = [subvolumes[k] for k in sorted(subvolumes.keys())]

        return scan

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
