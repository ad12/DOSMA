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

__all__ = ["CubeQuant"]

__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4
__R_SQUARED_THRESHOLD__ = 0.9
__INITIAL_T1_RHO_VAL__ = 70.0

__T1_RHO_LOWER_BOUND__ = 0.0
__T1_RHO_UPPER_BOUND__ = 500.0
__T1_RHO_DECIMAL_PRECISION__ = 3

_logger = logging.getLogger(__name__)


class CubeQuant(NonTargetSequence):
    """CubeQuant MRI sequence.

    Cubequant is a 3D fast-spin-echo (FSE) :math:`T_{1\\rho}`-weighted sequence.
    Acquisitions between spin-locks are susceptible to motion, and as a result,
    volumes within the scan have to be registered to each other (i.e. intra-registered).
    Intra-registration across different spin-locks is done by default upon construction.

    Moreover, CubeQuant scans often have lower resolution to increase SNR in practice.
    Because of the low-resolution, these scans are often registered to higher resolution
    target scans. This can be done using :meth:`CubeQuant.interregister`.
    """

    NAME = "cubequant"

    def __init__(self, volumes: Sequence[MedicalVolume], spin_lock_times: Sequence[float] = None):
        super().__init__(volumes=volumes)

        if spin_lock_times is None:
            try:
                if all(x.headers() is not None for x in self.volumes):
                    spin_lock_times = [x.get_metadata("EchoTime", float) for x in self.volumes]
            except (KeyError, AttributeError, RuntimeError) as e:
                raise ValueError(
                    f"Could not extract spin lock times from header. "
                    f"Please specify `spin_lock_times` argument - {e}"
                )
        self.spin_lock_times = spin_lock_times

    def intraregister(self):
        """Intra-register volumes.

        Patient could have moved between acquisition of different volumes,
        so different volumes of CubeQuant scan have to be registered with each other.

        The first spin lock time has the highest SNR, so it is used as the target.
        Volumes corresponding to the other spin lock times are registered to the target.

        Affine registration is done using Elastix.
        """
        self.__intraregister__()

    def interregister(self, target_path: str, target_mask_path: str = None):
        volumes = self.volumes
        spin_lock_times = self.spin_lock_times
        idxs = np.argsort(spin_lock_times)

        spin_lock_times = [spin_lock_times[i] for i in idxs]
        volumes = [volumes[i] for i in idxs]
        nr = NiftiReader()
        out_path = os.path.join(self.temp_path, "interregistered")
        os.makedirs(out_path, exist_ok=True)

        # TODO: Make these into parameters
        num_threads = 2
        num_workers = 0
        verbose = True

        base_image = volumes[0]
        moving = volumes[1:]

        if verbose:
            _logger.info("")
            _logger.info("==" * 40)
            _logger.info("Interregistering...")
            _logger.info("Target: {}".format(target_path))
            if target_mask_path is not None:
                _logger.info("Mask: {}".format(target_mask_path))
            _logger.info("==" * 40)

        if not target_mask_path:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
            use_mask = None
        else:
            target_mask_path = self.__dilate_mask__(target_mask_path, out_path)
            parameter_files = [
                fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE,
                fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE,
            ]
            use_mask = [False, True]

        out_reg, _ = register(
            target_path,
            base_image,
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
            show_pbar=True,
        )
        out_reg = out_reg[0]

        reg_vols = [nr.load(out_reg.warped_file)]
        for mvg in moving:
            reg_vols.append(apply_warp(mvg, out_reg.transform))

        # Undo sorting by spin lock time.
        reverse_idxs = {v: i for i, v in enumerate(idxs)}
        reg_vols = [reg_vols[reverse_idxs[k]] for k in sorted(reverse_idxs.keys())]

        self.volumes = reg_vols

    def generate_t1_rho_map(self, tissue: Tissue, mask_path: str = None, num_workers: int = 0):
        """
        Generate 3D T1-rho map and r-squared fit map using mono-exponential
        fit across subvolumes acquired at different spin lock times.

        Args:
            tissue (Tissue): Tissue to generate quantitative value for.
            mask_path (:obj:`str`, optional): File path to mask of ROI to analyze.
                If specified, only voxels specified by mask will be fit.
                This can considerably speeds up computation.
            num_workers (int, optional): Number of subprocesses to use for fitting.
                If `0`, will execute on the main thread.

        Returns:
            qv.T1Rho: T1-rho fit for tissue.

        Raises:
            ValueError: If `mask_path` specifies non-binary volume.
        """
        spin_lock_times = self.spin_lock_times
        subvolumes_list = self.volumes

        # only calculate for focused region if a mask is available, this speeds up computation
        mask = tissue.get_mask()
        if mask_path is not None:
            mask = (
                fio_utils.generic_load(mask_path, expected_num_volumes=1)
                if isinstance(mask_path, (str, os.PathLike))
                else mask_path
            )

        mef = MonoExponentialFit(
            bounds=(__T1_RHO_LOWER_BOUND__, __T1_RHO_UPPER_BOUND__),
            tc0="polyfit",
            decimal_precision=__T1_RHO_DECIMAL_PRECISION__,
            num_workers=num_workers,
            verbose=True,
        )

        t1rho_map, r2 = mef.fit(spin_lock_times, subvolumes_list, mask=mask)

        quant_val_map = qv.T1Rho(t1rho_map)
        quant_val_map.add_additional_volume("r2", r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def __intraregister__(self):
        """Intra-register volumes.

        Patient could have moved between acquisition of different volumes,
        so different volumes of CubeQuant scan have to be registered with each other.

        The first spin lock time has the highest SNR, so it is used as the target.
        Volumes corresponding to the other spin lock times are registered to the target.

        Affine registration is done using Elastix.
        """
        volumes = self.volumes
        spin_lock_times = self.spin_lock_times
        idxs = np.argsort(spin_lock_times)

        spin_lock_times = [spin_lock_times[i] for i in idxs]
        volumes = [volumes[i] for i in idxs]
        out_path = os.path.join(self.temp_path, "interregistered")
        os.makedirs(out_path, exist_ok=True)

        # TODO: Make these into parameters
        num_threads = 2
        num_workers = 0
        verbose = True

        if verbose:
            _logger.info("")
            _logger.info("==" * 40)
            _logger.info("Intraregistering...")
            _logger.info("==" * 40)

        out_path = os.path.join(self.temp_path, "intraregister")
        _, reg_vols = register(
            volumes[0],
            volumes[1:],
            fc.ELASTIX_AFFINE_PARAMS_FILE,
            out_path,
            num_workers=num_workers,
            num_threads=num_threads,
            return_volumes=True,
            rtype=tuple,
            show_pbar=verbose,
        )
        reg_vols = [volumes[0]] + list(reg_vols)

        # Transfer header information
        reg_vols = [
            reg._partial_clone(volume=False, headers=vol.headers())
            for (reg, vol) in zip(reg_vols, volumes)
        ]

        # Undo sorting by spin lock time.
        reverse_idxs = {v: i for i, v in enumerate(idxs)}
        reg_vols = [reg_vols[reverse_idxs[k]] for k in sorted(reverse_idxs.keys())]

        self.volumes = reg_vols

    def _save(self, metadata, save_dir: str, fname_fmt=None, **kwargs):
        default_fmt = {MedicalVolume: "echo-{}"}
        default_fmt.update(fname_fmt if fname_fmt else {})
        return super()._save(metadata, save_dir, fname_fmt=default_fmt, **kwargs)

    @classmethod
    def from_dict(cls, data, force: bool = False) -> "CubeQuant":
        interregistered_dirpath = None
        if "subvolumes" in data:
            interregistered_dirpath = os.path.dirname(data.pop("subvolumes")[0])
        scan: CubeQuant = super().from_dict(data, force=force)
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
        intraregister_action = ActionWrapper(
            name=cls.intraregister.__name__, help="register volumes within this scan"
        )
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
        generate_t1rho_map_action = ActionWrapper(
            name=cls.generate_t1_rho_map.__name__,
            help="generate T1-rho map",
            aliases=["t1_rho"],
            param_help={
                "mask_path": "Mask used for fitting select voxels - in nifti format (.nii.gz)"
            },
        )

        return [
            (cls.intraregister, intraregister_action),
            (cls.interregister, interregister_action),
            (cls.generate_t1_rho_map, generate_t1rho_map_action),
        ]
