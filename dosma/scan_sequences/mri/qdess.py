"""Quantitative Double Echo in Steady State (qDESS) sequence."""
import logging
import math
import warnings
from copy import deepcopy
from typing import Sequence, Tuple

import numpy as np
import pydicom
from pydicom.tag import Tag

from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import T2
from dosma.models.seg_model import SegModel
from dosma.scan_sequences.scans import ScanSequence
from dosma.tissues.tissue import Tissue
from dosma.utils.cmd_line_utils import ActionWrapper

__all__ = ["QDess"]

_logger = logging.getLogger(__name__)


class QDess(ScanSequence):
    """qDESS MRI sequence.

    Quantitative double echo in steady state (qDESS) is a high-resolution scan that consists
    of two echos (S1, S2) that has shown high efficacy for analytic :math:`T_2` mapping.
    Because of its high resolution, qDESS has been shown to be a good candidate for automatic
    segmentation.

    DOSMA supports both automatic segmentation and analytical T2 solving for qDESS scans.
    Automated segmentation uses pre-trained convolutional neural networks (CNNs).

    References:
        B Sveinsson, AS Chaudhari, GE Gold, BA Hargreaves. A simple analytic method
        for estimating :math:`T_2` in the knee from DESS. Magnetic Resonance in Medicine,
        38:63-70 (2017). `[link] <https://www.ncbi.nlm.nih.gov/pubmed/28017730>`_
    """

    NAME = "qdess"

    # DESS DICOM header keys
    __GL_AREA_TAG__ = Tag(0x001910B6)
    __TG_TAG__ = Tag(0x001910B7)

    # DESS constants
    __NUM_ECHOS__ = 2
    __VOLUME_DIMENSIONS__ = 3

    def __init__(self, volumes: Sequence[MedicalVolume]):
        if len(volumes) != 2:
            raise ValueError("QDess currently only supports 2 volumes.")
        super().__init__(volumes)

    def __validate_scan__(self) -> bool:
        """Validate that the dicoms are of qDESS sequence.

        Returns:
            bool: `True` if has 2 echos, `False` otherwise.
        """
        return len(self.volumes) == self.__NUM_ECHOS__

    def segment(self, model: SegModel, tissue: Tissue, use_rss: bool = False):
        """Segment tissue in scan.

        Args:
            model (SegModel): Model to use for segmenting scans.
            tissue (Tissue): The tissue to segment.
            use_rss (`bool`, optional): If ``True``, use root-sum-of-squares (RSS) of
                echos for segmentation (preferred for built-in methods).
                If ``False``, use first echo for segmentation.

        Returns:
            MedicalVolume: Binary mask for segmented region.
        """
        tissue_names = (
            ", ".join([t.FULL_NAME for t in tissue])
            if isinstance(tissue, Sequence)
            else tissue.FULL_NAME
        )
        _logger.info(f"Segmenting {tissue_names}...")

        if use_rss:
            segmentation_volume = self.calc_rss()
        else:
            # Use first echo for segmentation.
            segmentation_volume = self.volumes[0]

        # Segment tissue and add it to list.
        mask = model.generate_mask(segmentation_volume)
        if isinstance(mask, dict):
            if not isinstance(tissue, Sequence):
                tissue = [tissue]
            for abbreviation, tis in zip([t.STR_ID for t in tissue], tissue):
                tis.set_mask(mask[abbreviation])
                self.__add_tissue__(tis)
        else:
            assert isinstance(tissue, Tissue)
            tissue.set_mask(mask)
            self.__add_tissue__(tissue)

        return mask

    def generate_t2_map(
        self,
        tissue: Tissue = None,
        suppress_fat: bool = False,
        suppress_fluid: bool = False,
        beta: float = 1.2,
        gl_area: float = None,
        tg: float = None,
        tr: float = None,
        te: float = None,
        alpha: float = None,
        diffusivity: float = 1.25e-9,
        t1: float = None,
        nan_bounds: Tuple[float, float] = (0, 100),
        nan_to_num: float = 0.0,
        decimals: int = 1,
    ):
        """Generate 3D T2 map.

        Spoiler amplitude (``gl_area``) and duration (``tg``) must be specified if dicom header
        does not contain relevant private tags. If dicom header is unavailable, ``tr``, ``te``,
        and ``alpha`` must also be specified.

        All array-like arguments must be the same dimensions as the echo 1 and echo 2 volumes.

        Args:
            tissue (Tissue, optional): Tissue to generate T2 map for.
                If provided, the resulting quantitative value map will
                be added to the list of quantitative values for the tissue.
            suppress_fat (`bool`, optional): Suppress fat region in T2 computation.
                This can help reduce noise.
            suppress_fluid (`bool`, optional): Suppress fluid region in T2 computation.
                Fluid-nulled image is calculated as ``S1 - beta*S2``.
            beta (float, optional): Beta value used for suppressing fluid.
            gl_area (float, optional): Spoiler area.
                Defaults to value in dicom private tag '0x001910b6'.
                Required if dicom header unavailable or private tag missing.
            tg (float, optional): Spoiler duration (in microseconds).
                Defaults to value in dicom private tag ``0x001910b7``.
                Required if dicom header unavailable or private tag missing.
            tr (float, optional): Repitition time (in milliseconds).
                Required if dicom header unavailable.
            te (float, optional): Echo time (in milliseconds).
                Required if dicom header unavailable.
            alpha (float or array-like): Flip angle in degrees.
                Required if dicom header unavailable.
            diffusivity (float or array-like): Estimated diffusivity.
            t1 (float or array-like): Estimated t1 in milliseconds.
                Defaults to ``tissue.T1_EXPECTED`` if ``tissue`` provided.
            nan_bounds (float or Tuple[float, float], optional): The closed interval
                (``[a, b]``) for valid t2 values. All values outside of this interval will
                be set to ``nan``.
            nan_to_num (float): Value to be used to fill NaN values. If ``None``, values
                will not be replaced.
            decimals (int): Number of decimal places to round to. If ``None``, values
                will not be rounded.

        Returns:
            qv.T2: T2 fit map.
        """

        if self.volumes is None:
            raise ValueError("volumes and ref_dicom fields must be initialized")

        if (
            self.get_metadata(self.__GL_AREA_TAG__, gl_area) is None
            or self.get_metadata(self.__TG_TAG__, tg) is None
        ):
            raise ValueError(
                "Dicom headers do not contain tags for `gl_area` and `tg`. Please input manually"
            )

        xp = self.volumes[0].device.xp
        ref_dicom = self.ref_dicom if self.ref_dicom is not None else pydicom.Dataset()

        r, c, num_slices = self.volumes[0].volume.shape
        subvolumes = self.volumes

        # Split echos
        echo_1 = subvolumes[0].volume
        echo_2 = subvolumes[1].volume

        # All timing in seconds
        TR = (float(ref_dicom.RepetitionTime) if tr is None else tr) * 1e-3
        TE = (float(ref_dicom.EchoTime) if te is None else te) * 1e-3
        Tg = (float(ref_dicom[self.__TG_TAG__].value) if tg is None else tg) * 1e-6
        T1 = (float(tissue.T1_EXPECTED) if t1 is None else t1) * 1e-3

        # Flip Angle (degree -> radians)
        alpha = float(ref_dicom.FlipAngle) if alpha is None else alpha
        alpha = math.radians(alpha)
        if np.allclose(math.sin(alpha / 2), 0):
            warnings.warn("sin(flip angle) is close to 0 - t2 map may fail.")

        GlArea = float(ref_dicom[self.__GL_AREA_TAG__].value) if gl_area is None else gl_area

        Gl = GlArea / (Tg * 1e6) * 100
        gamma = 4258 * 2 * math.pi  # Gamma, Rad / (G * s).
        dkL = gamma * Gl * Tg

        # Simply math
        k = (
            xp.power((xp.sin(alpha / 2)), 2)
            * (1 + xp.exp(-TR / T1 - TR * xp.power(dkL, 2) * diffusivity))
            / (1 - xp.cos(alpha) * xp.exp(-TR / T1 - TR * xp.power(dkL, 2) * diffusivity))
        )

        c1 = (TR - Tg / 3) * (xp.power(dkL, 2)) * diffusivity

        # T2 fit
        mask = xp.ones([r, c, num_slices])

        ratio = mask * echo_2 / echo_1
        ratio = xp.nan_to_num(ratio)

        # have to divide division into steps to avoid overflow error
        t2map = -2000 * (TR - TE) / (xp.log(abs(ratio) / k) + c1)

        t2map = xp.nan_to_num(t2map)

        # Filter calculated T2 values that are below 0ms and over 100ms
        if nan_bounds is not None:
            lower, upper = nan_bounds
            t2map[(t2map < lower) | (t2map > upper)] = xp.nan
        if nan_to_num is not None:
            t2map = (
                xp.nan_to_num(t2map)
                if isinstance(nan_to_num, bool)
                else xp.nan_to_num(t2map, nan=nan_to_num)
            )

        if decimals is not None:
            t2map = xp.around(t2map, decimals)

        if suppress_fat:
            t2map = t2map * (echo_1 > 0.15 * xp.max(echo_1))

        if suppress_fluid:
            vol_null_fluid = echo_1 - beta * echo_2
            t2map = t2map * (vol_null_fluid > 0.1 * xp.max(vol_null_fluid))

        t2_map_wrapped = subvolumes[0]._partial_clone(volume=t2map, headers=True)
        t2_map_wrapped = T2(t2_map_wrapped)

        if tissue is not None:
            tissue.add_quantitative_value(t2_map_wrapped)

        return t2_map_wrapped

    def calc_rss(self):
        """Calculate root-sum-of-squares (RSS) of two echos.

        Returns:
            MedicalVolume: Volume corresponding to RSS of two echos.
        """
        return self._combine_echoes("rss")

    def _combine_echoes(self, method="rss"):
        """Combine two echoes.

        Args:
            method (str, optional): Supported combination methods are:
                * ``"rss"`` (default): The root-sum-of-squares of two echoes.
                * ``"rms"``: The root-mean-square of two echoes.

        Returns:
            MedicalVolume: Volume with RMS of two echos.
        """
        xp = self.volumes[0].device.xp

        if self.volumes is None:
            raise ValueError("Volumes must be initialized")

        assert len(self.volumes) == 2, "2 Echos expected"

        echo1 = xp.asarray(self.volumes[0].volume, dtype=xp.float64)
        echo2 = xp.asarray(self.volumes[1].volume, dtype=xp.float64)

        assert (~xp.iscomplex(echo1)).all() and (~xp.iscomplex(echo2)).all()

        if method == "rss":
            vol = xp.sqrt(echo1 ** 2 + echo2 ** 2)
        elif method == "rms":
            vol = xp.sqrt((echo1 ** 2 + echo2 ** 2) / 2)
        else:
            raise ValueError(f"`method={method}` is not supported")

        mv = deepcopy(self.volumes[0])
        mv.volume = vol

        return mv

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

        segment_action = ActionWrapper(
            name=cls.segment.__name__,
            help="generate automatic segmentation",
            param_help={"use_rss": "use root sum of squares (RSS) of two echos for segmentation"},
            alternative_param_names={"use_rss": ["rss"]},
        )
        generate_t2_map_action = ActionWrapper(
            name=cls.generate_t2_map.__name__,
            aliases=["t2"],
            param_help={
                "suppress_fat": "suppress computation on low SNR fat regions",
                "suppress_fluid": "suppress computation on fluid regions",
                "beta": "constant for calculating fluid-nulled image (S1-beta*S2)",
                "gl_area": "GL Area. Defaults to value in dicom tag '0x001910b6'",
                "tg": "Gradient time (in microseconds). "
                "Defaults to value in dicom tag '0x001910b7'.",
                "alpha": "Flip angle in degrees. Defaults to value in dicom tag '0x00181314'.",
                "diffusivity": "Estimated diffusivity. Defaults to 1.25e-9",
            },
            help="generate T2 map",
        )

        return [(cls.segment, segment_action), (cls.generate_t2_map, generate_t2_map_action)]
