import logging
import os
from copy import deepcopy
from typing import List, Sequence

from nipype.interfaces.elastix import Registration

from dosma import file_constants as fc
from dosma.core import quant_vals as qv
from dosma.core.fitting import MonoExponentialFit
from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader
from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import QuantitativeValueType
from dosma.defaults import preferences
from dosma.scan_sequences.scans import ScanSequence
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils
from dosma.utils.cmd_line_utils import ActionWrapper

__all__ = ["Mapss"]

__EXPECTED_NUM_ECHO_TIMES__ = 7

__INITIAL_T1_RHO_VAL__ = 70.0
__T1_RHO_LOWER_BOUND__ = 0
__T1_RHO_UPPER_BOUND__ = 500

__INITIAL_T2_VAL__ = 30.0
__T2_LOWER_BOUND__ = 0
__T2_UPPER_BOUND__ = 100

__DECIMAL_PRECISION__ = 3

_logger = logging.getLogger(__name__)


class Mapss(ScanSequence):
    """MAPSS MRI sequence.

    Magnetization‐prepared angle‐modulated partitioned k‐space spoiled gradient echo snapshots
    (3D MAPSS) is a spoiled gradient (SPGR) sequence that reduce specific absorption rate (SAR),
    increase SNR, and reduce the extent of retrospective correction of contaminating T1 effects.

    The MAPSS sequence can be used to estimate both T1𝜌 and T2 quantitative values.
    MAPSS scans must also be intra-registered to ensure alignment between all volumes
    acquired at different echos and spin-lock times. Intra-registration is performed
    automatically upon construction. :math:`T_2` and :`T_{1\\rho}` fitting is also supported.

    References:
        X Li, ET Han, RF Busse, S Majumdar. In vivo t1ρ mapping in
        cartilage using 3d magnetization-prepared angle-modulated partitioned k-space spoiled
        gradient echo snapshots (3d mapss). Magnetic Resonance in Medicine, 59(2):298–307 (2008).
    """

    NAME = "mapss"

    def __init__(self, volumes: Sequence[MedicalVolume], echo_times: Sequence[float] = None):
        if not isinstance(volumes, Sequence):
            raise ValueError("`volumes` must be sequence of MedicalVolumes.")

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

    def __validate_scan__(self):
        return len(self.volumes) == 7

    def __intraregister__(self, volumes: List[MedicalVolume]):
        """Intraregister volumes.

        Sets `self.volumes` to intraregistered volumes.

        Args:
            volumes (list[MedicalVolume]): Volumes to register.

        Raises:
            TypeError: If `volumes` is not `list[MedicalVolume]`.
        """
        if (
            (not volumes)
            or (type(volumes) is not list)
            or (len(volumes) != __EXPECTED_NUM_ECHO_TIMES__)
        ):
            raise TypeError("`volumes` must be of type List[MedicalVolume]")

        num_echos = len(volumes)

        _logger.info("")
        _logger.info("==" * 40)
        _logger.info("Intraregistering...")
        _logger.info("==" * 40)

        # temporarily save subvolumes as nifti file
        raw_volumes_base_path = io_utils.mkdirs(os.path.join(self.temp_path, "raw"))

        # Use first subvolume as a basis for registration
        # Save in nifti format to use with elastix/transformix
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
            reg.inputs.output_path = io_utils.mkdirs(
                os.path.join(self.temp_path, "intraregistered", "{:03d}".format(echo_index))
            )
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            reg.terminal_output = preferences.nipype_logging
            _logger.info("Registering {} -> {}".format(str(echo_index), str(target_echo_index)))
            tmp = reg.run()

            warped_file = tmp.outputs.warped_file
            intrareg_vol = nr.load(warped_file)

            # copy affine from original volume, because nifti changes loading accuracy
            intrareg_vol = MedicalVolume(
                volume=intrareg_vol.volume,
                affine=volumes[echo_index].affine,
                headers=deepcopy(volumes[echo_index].headers()),
            )

            intraregistered_volumes.append(intrareg_vol)

        self.volumes = intraregistered_volumes

    def intraregister(self):
        """Intra-register volumes."""
        self.__intraregister__(self.volumes)

    def generate_t1_rho_map(
        self, tissue: Tissue = None, mask_path: str = None, num_workers: int = 0
    ):
        """
        Generate 3D T1-rho map and r-squared fit map using mono-exponential fit
        across subvolumes acquired at different echo times.

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
        tc0 = "polyfit"
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(
            qv.T1Rho, echo_inds, tissue, bounds, tc0, decimal_precision, mask_path, num_workers
        )

        return qv_map

    def generate_t2_map(self, tissue: Tissue = None, mask_path: str = None, num_workers: int = 0):
        """
        Generate 3D T2 map and r-squared fit map using mono-exponential fit
        across subvolumes acquired at different echo times.

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
        tc0 = "polyfit"
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(
            qv.T2, echo_inds, tissue, bounds, tc0, decimal_precision, mask_path, num_workers
        )

        return qv_map

    def __fitting_helper(
        self,
        qv_type: QuantitativeValueType,
        echo_inds: Sequence[int],
        tissue: Tissue,
        bounds,
        tc0,
        decimal_precision,
        mask_path,
        num_workers,
    ):
        echo_info = [(self.echo_times[i], self.volumes[i]) for i in echo_inds]

        # sort by echo time
        echo_info = sorted(echo_info, key=lambda x: x[0])

        xs = [et for et, _ in echo_info]
        ys = [vol for _, vol in echo_info]

        # only calculate for focused region if a mask is available, this speeds up computation
        mask = tissue.get_mask()
        if mask_path is not None:
            mask = (
                fio_utils.generic_load(mask_path, expected_num_volumes=1)
                if isinstance(mask_path, (str, os.PathLike))
                else mask_path
            )

        mef = MonoExponentialFit(
            bounds=bounds,
            tc0=tc0,
            decimal_precision=decimal_precision,
            num_workers=num_workers,
            verbose=True,
        )
        qv_map, r2 = mef.fit(xs, ys, mask=mask)

        quant_val_map = qv_type(qv_map)
        quant_val_map.add_additional_volume("r2", r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def _save(self, metadata, save_dir, fname_fmt=None, **kwargs):
        default_fmt = {MedicalVolume: "echo-{}"}
        default_fmt.update(fname_fmt if fname_fmt else {})
        return super()._save(metadata, save_dir, fname_fmt=default_fmt, **kwargs)

    @classmethod
    def cmd_line_actions(cls):
        """
        Provide command line information (such as name, help strings, etc)
        as list of dictionary.
        """
        intraregister_action = ActionWrapper(
            name=cls.intraregister.__name__, help="register volumes within this scan"
        )
        generate_t1_rho_map_action = ActionWrapper(
            name=cls.generate_t1_rho_map.__name__,
            aliases=["t1_rho"],
            param_help={
                "mask_path": (
                    "mask filepath (.nii.gz) to reduce computational time for fitting. "
                    "Not required if loading data (ie. `--l` flag) for tissue with mask."
                )
            },
            alternative_param_names={"mask_path": ["mask", "mp"]},
            help="generate T1-rho map using mono-exponential fitting",
        )
        generate_t2_map_action = ActionWrapper(
            name=cls.generate_t2_map.__name__,
            aliases=["t2"],
            param_help={
                "mask_path": (
                    "mask filepath (.nii.gz) to reduce computational time for fitting. "
                    "Not required if loading data (ie. `--l` flag) for tissue with mask."
                )
            },
            alternative_param_names={"mask_path": ["mask", "mp"]},
            help="generate T2 map using mono-exponential fitting",
        )

        return [
            (cls.intraregister, intraregister_action),
            (cls.generate_t1_rho_map, generate_t1_rho_map_action),
            (cls.generate_t2_map, generate_t2_map_action),
        ]
