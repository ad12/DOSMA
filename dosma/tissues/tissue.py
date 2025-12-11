import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import scipy.ndimage as sni

from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import SAGITTAL
from dosma.core.quant_vals import QuantitativeValue, QuantitativeValueType
from dosma.defaults import preferences
from dosma.utils import io_utils

WEIGHTS_FILE_EXT = "h5"

__all__ = ["Tissue"]


class Tissue(ABC):
    """Abstract class for tissues.

    Tissues are defined loosely as any tissue structures (bones, soft tissue, etc.).

    Args:
        weights_dir (str): Directory to all segmentation weights.
        medial_to_lateral (`bool`, optional): If `True`, anatomy is from medial_to_lateral.

    Attributes:
        FULL_NAME (str): Full name of tissue 'femoral cartilage' for femoral cartilage.
        ID (int): Unique integer ID for tissue. Should be unique to all tissues,
            and should not change.
        STR_ID (str): Short hand string id such as 'fc' for femoral cartilage.
        T1_EXPECTED (float): Expected T1 value (in milliseconds).
        medial_to_lateral (bool): If ``True``, mask is in medial to lateral direction.
        pid (str): Patient/subject ID. Should be anonymized.
        quant_vals (dict[str, tuple[np.ndarray, pd.DataFrame]]): Mapping from quantitative value
            name (t2, t1-rho, etc.) to tuple of unrolled map and DataFrame containing
            measurement values.
        weights_filepath (str): File path to weights directory for neural network segmentation.
    """

    ID = -1
    STR_ID = ""
    FULL_NAME = ""

    # Expected quantitative param values.
    T1_EXPECTED = None

    def __init__(self, weights_dir: str = None, medial_to_lateral: bool = None):
        self.pid = None
        self.__mask__ = None
        self.quant_vals = {}
        self.weights_file_path = None

        if weights_dir is not None:
            self.weights_file_path = self.find_weights(weights_dir)

        self.medial_to_lateral = medial_to_lateral

        # quantitative value list
        self.quantitative_values = []

    @abstractmethod
    def split_regions(self, base_map: Union[np.ndarray, MedicalVolume]):
        """Split mask into anatomical regions.

        Args:
            base_map (np.ndarray): 3D numpy array typically corresponding to volume to split.

        Returns:
            np.ndarray: 4D numpy array (region, height, width, depth).
                        Saved in variable `self.regions`.
        """
        pass

    def calc_quant_vals(self):
        """Calculate quantitative values for pixels corresponding to the tissue.

        Requires mask to be set for this tissue.
        """
        for qv in self.quantitative_values:
            self.__calc_quant_vals__(qv.volumetric_map, qv.qv_type)

    @abstractmethod
    def __calc_quant_vals__(self, quant_map: MedicalVolume, map_type: QuantitativeValueType):
        """Helper method to get quantitative values for tissue - implemented per tissue.

        Different tissues should override this as they see fit.

        Args:
            quant_map (MedicalVolume): 3D map of pixel-wise quantitative measures
                (T2, T2*, T1-rho, etc.). Volume should have ``np.nan`` values for
                all pixels unable to be calculated.
            map_type (QuantitativeValueType): Type of quantitative value to analyze.

        Raises:
            TypeError: If `quant_map` is not of type `MedicalVolume` or `map_type` is not of type
                `QuantitativeValueType`.
            ValueError: If no mask is found for tissue.
        """
        if not isinstance(quant_map, MedicalVolume):
            raise TypeError("`Expected type 'MedicalVolume' for `quant_map`")
        if not isinstance(map_type, QuantitativeValueType):
            raise TypeError("`Expected type 'QuantitativeValueType' for `map_type`")

        if self.__mask__ is None:
            raise ValueError("Please initialize mask for {}".format(self.FULL_NAME))

        quant_map.reformat(self.__mask__.orientation, inplace=True)
        pass

    def __store_quant_vals__(
        self, quant_map: MedicalVolume, quant_df: pd.DataFrame, map_type: QuantitativeValueType
    ):
        """Adds quantitative value in `self.quant_vals`.

        Args:
            quant_map (list[dict]): Dictionaries of different unrolled maps and
                corresponding plotting data (title, xlabel, etc.).
            quant_df (pd.DataFrame): Computed data for this quantitative value.
            map_type (QuantitativeValueType): Type of quantitative value to analyze.
        """
        self.quant_vals[map_type.name] = (quant_map, quant_df)

    def find_weights(self, weights_dir: str):
        """Search for weights file in weights directory.

        Args:
            weights_dir (str): Directory where weights are stored.

        Returns:
            str: File path to weights corresponding to tissue.

        Raises:
            ValueError: If multiple weights files exists for the tissue
                or no valid weights file found.
        """

        # Find weights file with NAME in the filename, like 'fc_weights.h5'
        files = os.listdir(weights_dir)
        weights_file = None
        for f in files:
            file = os.path.join(weights_dir, f)
            if os.path.isfile(file) and f.endswith(WEIGHTS_FILE_EXT) and self.STR_ID in f:
                if weights_file is not None:
                    raise ValueError("There are multiple weights files, please remove duplicates")
                weights_file = file

        if weights_file is None:
            raise ValueError(
                "No file found that contains '{}' and ends in '{}'".format(
                    self.STR_ID, WEIGHTS_FILE_EXT
                )
            )

        self.weights_file_path = weights_file

        return weights_file

    def save_data(
        self, save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format
    ):
        """Save data for tissue.

        Saves mask and quantitative values associated with this tissue.

        Override in subclasses to save additional data. When overriding in subclasses, call
        ``super().save_data(save_dirpath)`` first to save mask and quantitative values by default.
        See :mod:`dosma.tissues.femoral_cartilage` for details.

        .. literalinclude:: femoral_cartilage.py

        Args:
            save_dirpath (str): Directory path where all data is stored.
            data_format (`ImageDataFormat`, optional): Format to save data.
        """
        save_dirpath = self.__save_dirpath__(save_dirpath)

        if self.__mask__ is not None:
            mask_file_path = os.path.join(save_dirpath, "{}.nii.gz".format(self.STR_ID))
            mask_file_path = fio_utils.convert_image_data_format(mask_file_path, data_format)
            self.__mask__.save_volume(mask_file_path, data_format=data_format)

        for qv in self.quantitative_values:
            qv.save_data(save_dirpath, data_format)

        self.__save_quant_data__(save_dirpath)

    @abstractmethod
    def __save_quant_data__(self, dirpath: str):
        """Save quantitative data generated for this tissue.

        Called by `save_data`.

        Args:
            dirpath (str): Directory path to tissue data.
        """
        pass

    def save_quant_data(self, dirpath: str):
        """Save quantitative data generated for this tissue.

        Does not save mask or quantitative parameter map.

        Args:
            dirpath (str): Directory path to tissue data.
        """
        return self.__save_quant_data__(dirpath)

    def load_data(self, load_dir_path: str):
        """Load data for tissue.

        All tissue information is based on the mask. If mask for tissue doesn't exist,
        there is no information to load.

        Args:
            load_dir_path (str): Directory path where all data is stored.
        """
        load_dir_path = self.__save_dirpath__(load_dir_path)
        mask_file_path = os.path.join(load_dir_path, "{}.nii.gz".format(self.STR_ID))

        # Try to load mask, if file exists.
        try:
            msk = fio_utils.generic_load(mask_file_path, expected_num_volumes=1)
            self.set_mask(msk)
        except FileNotFoundError:
            # do nothing
            pass

        self.quantitative_values = QuantitativeValue.load_qvs(load_dir_path)

    def __save_dirpath__(self, dirpath):
        """Tissue-specific subdirectory to store data.

        Subdirectory will have path '`dirpath`/`self.STR_ID`/'.

        If directory does not exist, it will be created.

        Args:
            dirpath (str): Directory path where all data is stored.

        Returns:
            str: Tissue-specific data directory.
        """
        return io_utils.mkdirs(os.path.join(dirpath, self.STR_ID))

    # TODO (arjundd): Refactor get/set methods of mask to property.
    def set_mask(self, mask: MedicalVolume):
        """Set mask for tissue.

        Args:
            mask (MedicalVolume): Binary mask of segmented tissue.
        """
        assert type(mask) is MedicalVolume, "mask for tissue must be of type MedicalVolume"
        mask = mask.reformat(SAGITTAL)
        self.__mask__ = mask

    def get_mask(self):
        """
        Returns:
            MedicalVolume: Binary mask of segmented tissue.
        """
        return self.__mask__

    def add_quantitative_value(self, qv_new: QuantitativeValue):
        """Add quantitative value to the tissue.

        Args:
            qv_new (QuantitativeValue): Quantitative value to add to tissue.
        """
        # for qv in self.quantitative_values:
        #     if qv_new.NAME == qv.NAME:
        #         raise ValueError('This quantitative value already exists. '
        #                          'Only one type of quantitative value can be added per tissue.\n'
        #                          'Manually delete %s folder' % qv_new.NAME)

        self.quantitative_values.append(qv_new)

    def __get_axis_bounds__(
        self, im: np.ndarray, ignore_nan: bool = True, leave_buffer: bool = False
    ):
        """Get tightest bounds for data in the array.

        When plotting data, we would like to avoid making our dynamic range too large such
        that we cannot detect color changes in differences that matter.
        To avoid this, we make our bounds as tight as possible.

        Bounds are calculated with respect to non-zero elements. If unique values are [0, 8, 9],
            the dyanmic range will be [8, 9].

        Args:
            im (np.ndarray): Array containing information for which bounds have to be computed.
            ignore_nan (obj:`bool`, optional): Ignore `nan` values when computing the bounds.
            leave_buffer (obj:`bool`, optional): Add buffer of +/-5 to dynamic range.
        """
        im_temp = im
        axs = []
        if ignore_nan:
            im_temp = np.nan_to_num(im)

        non_zero_elems = np.nonzero(im_temp)

        for i in range(len(non_zero_elems)):
            v_min = np.min(non_zero_elems[i])
            v_max = np.max(non_zero_elems[i])
            if leave_buffer:
                v_min -= 5
                v_max += 5

            axs.append((v_min, v_max))

        return axs


def largest_cc(mask, num=1):
    """Return the largest `num` connected component(s) of a 3D mask array.

    Args:
        mask (np.ndarray): 3D mask array (`bool` or `np.[u]int`).
        num (int, optional): Maximum number of connected components to keep.

    Returns:
        mask (np.ndarray): 3D mask array with `num` connected components.


    Note:
        Adapted from nipy (https://github.com/nipy/nipy/blob/master/nipy/labs/mask.py)
        due to dependency issues.
    """
    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = sni.label(mask)
    if not label_nb:
        raise ValueError("No non-zero values: no connected components")
    if label_nb == 1:
        return mask.astype(bool)
    label_count = np.bincount(labels.ravel().astype(int))
    # discard 0 the 0 label
    label_count[0] = 0

    # Split num=1 case for speed.
    if num == 1:
        return labels == label_count.argmax()
    else:
        # 1) discard 0 the 0 label and 2) descending order
        order = np.argsort(label_count)[1:][::-1]
        return np.isin(labels, order[:num])
