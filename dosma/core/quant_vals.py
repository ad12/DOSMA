import os
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences

__all__ = ["QuantitativeValueType", "QuantitativeValue", "T1Rho", "T2", "T2Star"]


class QuantitativeValueType(Enum):
    """Enum of types of quantitative values that can be analyzed.

    For more information on quantitative parameters, see :obj:`QuantitativeValue`.
    """

    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


class QuantitativeValue(ABC):
    """This class handles tracking volumes associated with different quantitative values.

    Quantitative MRI characterizes the relaxation profile of different regions in the volume.
    This profile is determined by the composition of the object and has been shown to be
    informative for early detection of pathology.

    In practice, many of these quantitative relaxation parameters
    (:math:`T_1`, :math:`T_2`, :math:`T_2^*`, etc.) are computed per voxel by fitting
    to the exponential decay/recovery curves or by some analytical method. This results
    in a volume where different voxels have different relaxation parameters. These volumes
    are referred to as *volumetric quantitative maps* or *quantitative maps*.

    These fitted/computed relaxation parameters are called quantitative values.

    Args:
        volumetric_map (MedicalVolume, optional): Volumetric quantitative map.

    Attributes:
        volumetric_map (MedicalVolume): Volumetric quantitative map.
        additional_volumes (Dict[str, MedicalVolume]): Additional volumes associated
            with quantitative value. These are typically volumes associated with the
            goodness of fit of the value. For example, a volume could be pixel-wise r-squared,
            or error bounds, etc.

    Note:
        This class is in alpha - signature and functionality changes are very likely.
    """

    ID = 0
    NAME = ""

    def __init__(self, volumetric_map: MedicalVolume = None):
        assert self.ID > 0, "Attribute `ID` not initialized for {}".format(type(self))
        assert self.NAME != "", "Attribute `NAME` not initialized for {}".format(type(self))

        # Main 3D quantitative value map (MedicalVolume)
        if volumetric_map is not None and not isinstance(volumetric_map, MedicalVolume):
            raise TypeError("`volumetric_map` must be of type MedicalVolume")

        self.volumetric_map = volumetric_map

        # Sometimes there are additional volumes that we want to keep track of
        # i.e. r2 values, goodness of fit, etc
        # Store these as string_name: MedicalVolume (i.e. "r2: MedicalVolume)
        # To add values to this field, use the add_additional_volume method below
        # these results will not be loaded
        self.additional_volumes = {}

    def save_data(
        self, dir_path: str, data_format: ImageDataFormat = preferences.image_data_format
    ):
        """Save data to disk.

        Data will be stored in folder '`dir_path`/`self.NAME`'.

        Args:
            dir_path (str): Directory path.
            data_format (:obj:`ImageDataFormat`, optional): Data format to save medical volumes.
                Defaults to ``preferences.image_data_format``.
        """
        if data_format != ImageDataFormat.nifti:
            import warnings

            warnings.warn(
                "Due to bit depth issues, only nifti format is supported for quantitative values. "
                "Writing as nifti file..."
            )
            data_format = ImageDataFormat.nifti

        if self.volumetric_map is not None:
            filepath = os.path.join(dir_path, self.NAME, "{}.nii.gz".format(self.NAME))
            # filepath = fio_utils.convert_format_filename(filepath, data_format)
            self.volumetric_map.save_volume(filepath, data_format=data_format)

        for volume_name in self.additional_volumes.keys():
            add_vol_filepath = os.path.join(
                dir_path, self.NAME, "{}-{}.nii.gz".format(self.NAME, volume_name)
            )
            # add_vol_filepath = fio_utils.convert_format_filename(add_vol_filepath, data_format)
            self.additional_volumes[volume_name].save_volume(
                add_vol_filepath, data_format=data_format
            )

    def load_data(self, dir_path):
        """Load data from disk.

        Data will be loaded from folder '`dir_path`/`self.NAME`'.

        Currently, additional volumes are not reloaded.

        Args:
            dir_path (str): Directory path.
        """
        file_path = os.path.join(dir_path, self.NAME, "{}.nii.gz".format(self.NAME))
        qv_volume = fio_utils.generic_load(file_path, expected_num_volumes=1)

        self.volumetric_map = qv_volume

    def add_additional_volume(self, name: str, volume: MedicalVolume):
        """Add volume that corresponds to quantitative value.

        Additional volumes are typically volumes associated with the goodness of fit of the value.
        For example, a volume could be r-squared values per voxel, or error bounds, etc.

        This should not be the volumetric quantitative map.
        To update that map, see ``self.volumetric_map``.

        Args:
            name (str): Name of additional volume.
            volume (MedicalVolume):
        """
        if not isinstance(volume, MedicalVolume):
            raise TypeError("`volumes` must be of type MedicalVolume")
        self.additional_volumes[name] = volume

    def to_metrics(
        self,
        mask: MedicalVolume = None,
        labels: Dict[int, str] = None,
        bounds: Tuple[float, float] = None,
        closed: str = "right",
        fns: Dict[str, Callable] = None,
    ) -> pd.DataFrame:
        """Compute scalar metrics for quantitative values.

        Metrics include mean, median, standard deviation, and number of voxels.
        Valid voxels are defined as finite valued voxels within the interval
        `bounds` (if specified).

        Args:
            mask (MedicalVolume, optional): Label mask. Labels should be unsigned ints (uint).
                Metrics will be computed for each non-zero label(s).
                If `labels` specified, metrics computed only for keys in `labels` dictionary.
                If not specified, metrics will be calculated over all valid voxels.
            labels (Dict[int, str], optional): Mapping from label to label name.
                If specified, only labels in this argument will be computed.
            bounds (float or Tuple[float, float], optional): The left/right bounds
                for computing metrics. By default, the bounds are the
                open-interval `(-inf, inf)`.
            closed (str, optional): If `bounds` specified, whether the bounds are closed
                on the left-side, right-side, both or neither.
                One of {'right', 'left', 'both', 'neither'}.

        Returns:
            metrics (pd.DataFrame): Metrics for quantitative value. Columns include:
                * "Region" (str): The label name.
                * "Mean" (float): Average quantitative value in a region.
                * "Median" (float): Median quantitative value in a region.
                * "Std" (float): Standard deviation of quantitative value in a region.
                * "# Voxels" (int): The number of valid voxels in the region.
        """
        volume = self.volumetric_map.volume
        valid_mask = np.isfinite(volume)
        if bounds:
            assert len(bounds) == 2, len(bounds)  # Expected (left,right) bound
            lb, ub = bounds[0], bounds[1]
            assert lb <= ub, f"lower:{lb}, upper: {ub}"  # Expected left bound <= right bound
            assert closed in ("right", "left", "both", "neither"), closed
            lb_mask = volume >= lb if closed in ("left", "both") else volume > lb
            ub_mask = volume <= ub if closed in ("right", "both") else volume < ub
            valid_mask &= lb_mask & ub_mask

        if mask is not None:
            mask = mask.reformat(self.volumetric_map.orientation)
            mask = mask.volume

            if labels is None:
                unique_vals = [x for x in np.unique(mask) if x > 0]
                labels = {int(i): f"label_{int(i)}" for i in unique_vals}
            labels.update({-1: "total"})
        else:
            labels = {-2: "total"}

        if mask is not None:
            # Do not modify original array.
            mask = mask.copy()
            mask[~valid_mask] = 0

        if fns is None:
            fns = {}

        metrics = defaultdict(list)
        for label, name in labels.items():
            if label == -2:
                qv_region_vals = volume[valid_mask]  # Entire volume.
            elif label == -1:
                qv_region_vals = volume[mask > 0]  # noqa
            else:
                qv_region_vals = volume[mask == label]
            num_voxels = np.prod(qv_region_vals.shape)

            metrics["Category"].append(name)
            metrics["Mean"].append(np.nanmean(qv_region_vals))
            metrics["Std"].append(np.nanstd(qv_region_vals))
            metrics["Median"].append(np.nanmedian(qv_region_vals))
            metrics["# Voxels"].append(num_voxels)
            for name, fn in fns.items():
                metrics[name].append(fn(qv_region_vals))

        return pd.DataFrame(metrics)

    def to(self, device):
        """Moves volumes of this scan onto the appropriate device.

        This is an inplace operation. A new ScanSequence will not
        be returned.

        Args:
            device: The device to move to.

        Returns:
            self
        """
        self.volumetric_map = self.volumetric_map.to(device)
        self.additional_volumes = {k: v.to(device) for k, v in self.additional_volumes.items()}

        return self

    @staticmethod
    def get_qv(qv_id: Union[int, str]):
        """Find QuantitativeValue enum using id or name.

        Args:
            qv_id (:obj:`int` or :obj:`str`): Either quantitative value enum number
                or name in lower case.

        Returns:
            QuantitativeValue: Quantitative value corresponding to id.

        Raises:
            ValueError: If no QuantitativeValue corresponding to `qv_id` found.
        """
        for qv in [T1Rho(), T2(), T2Star()]:
            if qv.NAME.lower() == qv_id or qv.NAME == qv_id or qv.ID == qv_id:
                return qv

        raise ValueError("Quantitative Value with name or id {} not found".format(qv_id))

    @staticmethod
    def save_qvs(dir_path: str, qvs):
        """Save all quantitative values from directory.

        Args:
            dir_path (str): Directory path.
            qvs (Sequence[QuantitativeValue]): Quantitative values to save.
        """
        for qv in qvs:
            if not isinstance(qv, QuantitativeValue):
                raise TypeError("All members of `qvs` must be instances of QuantitativeValue")
            qv.save_data(dir_path)

    @staticmethod
    def load_qvs(dir_path: str):
        """Load all quantitative values from directory.

        Args:
            dir_path (str): Directory path.

        Returns:
            list[QuantitativeValue]: Quantitative value wrappers.
        """
        qvs = []
        for qv in [T1Rho(), T2(), T2Star()]:
            possible_qv_filepath = os.path.join(dir_path, qv.NAME, "{}.nii.gz".format(qv.NAME))
            if os.path.isfile(possible_qv_filepath):
                qv.load_data(dir_path)
                qvs.append(qv)

        return qvs

    @property
    def qv_type(self) -> QuantitativeValueType:
        """QuantitativeValueType: quantitative value type."""
        raise NotImplementedError(f"Quantitative value type not implemented for {type(self)}")


class T1Rho(QuantitativeValue):
    """T1Rho MRI parameter."""

    ID = 1
    NAME = "t1_rho"

    @property
    def qv_type(self):
        return QuantitativeValueType.T1_RHO


class T2(QuantitativeValue):
    """T2 MRI parameter."""

    ID = 2
    NAME = "t2"

    @property
    def qv_type(self):
        return QuantitativeValueType.T2


class T2Star(QuantitativeValue):
    """T2Star MRI parameter."""

    ID = 3
    NAME = "t2_star"

    @property
    def qv_type(self):
        return QuantitativeValueType.T2_STAR
