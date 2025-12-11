"""Analysis for patellar cartilage.

Attributes:
    BOUNDS (dict): Upper bounds for quantitative values.
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import scipy

if scipy.__version__ >= "2.0.0":
    from scipy.ndimage import center_of_mass as c_of_m
else:
    from scipy.ndimage.measurements import center_of_mass as c_of_m

from dosma.core.device import get_array_module
from dosma.core.quant_vals import QuantitativeValueType
from dosma.defaults import preferences
from dosma.tissues.tissue import Tissue, largest_cc
from dosma.utils import io_utils

import matplotlib.pyplot as plt

# milliseconds
BOUNDS = {
    QuantitativeValueType.T2: 60.0,
    QuantitativeValueType.T1_RHO: 100.0,
    QuantitativeValueType.T2_STAR: 50.0,
}

__all__ = ["PatellarCartilage"]


class PatellarCartilage(Tissue):
    """Handles analysis and visualization for patellar cartilage."""

    ID = 3
    STR_ID = "pc"
    FULL_NAME = "patellar cartilage"

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    # Region Keys
    _ANTERIOR_KEY = 0
    _POSTERIOR_KEY = 1
    _CORONAL_KEYS = [_ANTERIOR_KEY, _POSTERIOR_KEY]

    _MEDIAL_KEY = 0
    _LATERAL_KEY = 1
    _SAGITTAL_KEYS = [_MEDIAL_KEY, _LATERAL_KEY]

    _REGION_DEEP_KEY = 0
    _REGION_SUPERFICIAL_KEY = 1
    _TOTAL_AXIAL_KEY = -1

    def __init__(self, weights_dir: str = None, medial_to_lateral: bool = None):
        super().__init__(weights_dir=weights_dir, medial_to_lateral=medial_to_lateral)

        self.regions_mask = None

    def unroll_coronal(self, quant_map: np.ndarray):
        """Unroll patellar cartilage in the coronal direction.

        Because patellar cartilage is flat, "unrolling" is projecting the patellar
        cartilage onto the coronal axis.

        Args:
            quant_map (np.ndarray):
        """
        mask = self.__mask__.volume

        assert (
            self.regions_mask is not None
        ), "region_mask not initialized. Should be initialized when mask is set"
        region_mask_deep_superficial = self.regions_mask[..., 0]

        superficial = (
            (region_mask_deep_superficial == self._REGION_SUPERFICIAL_KEY) * mask * quant_map
        )
        superficial[superficial == 0] = np.nan
        superficial = np.nanmean(superficial, axis=2)

        deep = (region_mask_deep_superficial == self._REGION_DEEP_KEY) * mask * quant_map
        deep[deep == 0] = np.nan
        deep = np.nanmean(deep, axis=2)

        total = mask * quant_map
        total[total == 0] = np.nan
        total = np.nanmean(total, axis=2)

        return total, superficial, deep

    def split_regions(self, base_map):
        """Split patellar cartilage into deep/superficial regions.

        For patellar cartilage, the superficial/deep transition occurs in
        the anterior/posterior (A/P) direction. The boundary is determined
        for each non-zero 1D column spanning independently by the local
        center-of-mass (COM). The medial/lateral (M/L) plane is computed
        using the global COM.

        Args:
            base_map (ndarray): Binary 3D mask with orientation (SI, AP, ML/LM).
                If `self.medial_to_lateral`, last dimension should be ML.
        """
        if np.sum(base_map) == 0:
            warnings.warn("No mask for `%s` was found." % self.FULL_NAME, stacklevel=2)

        # Superficial/Deep (A/P)
        locs = base_map.sum(axis=1).nonzero()
        voxels = base_map[locs[0], :, locs[1]]
        com_sup_inf = np.asarray(
            [int(np.ceil(c_of_m(voxels[i, :])[0])) for i in range(voxels.shape[0])]
        )
        region_mask_sup_deep = np.full(base_map.shape, self._REGION_DEEP_KEY)
        for i in range(len(com_sup_inf)):
            region_mask_sup_deep[
                locs[0][i], : com_sup_inf[i], locs[1][i]
            ] = self._REGION_SUPERFICIAL_KEY

        # M/L
        cum_ml = np.nonzero(base_map.sum(axis=(0, 1)))[0]  # noqa: F841
        # midpoint_ml = int(np.ceil((np.min(cum_ml) + np.max(cum_ml)) / 2))
        midpoint_ml = int(np.ceil(c_of_m(base_map)[2]))
        region_mask_med_lat = np.full(base_map.shape, self._LATERAL_KEY)
        medial_span = slice(0, midpoint_ml) if self.medial_to_lateral else slice(midpoint_ml, None)
        region_mask_med_lat[:, :, medial_span] = self._MEDIAL_KEY

        self.regions_mask = np.stack([region_mask_sup_deep, region_mask_med_lat], axis=-1)

    def __calc_quant_vals__(self, quant_map, map_type):
        subject_pid = self.pid

        super().__calc_quant_vals__(quant_map, map_type)

        assert (
            self.regions_mask is not None
        ), "region_mask not initialized. Should be initialized when mask is set"

        quant_map_volume = quant_map.volume
        mask = self.__mask__.volume
        quant_map_volume = mask * quant_map_volume

        deep_superficial_map = self.regions_mask[..., 0]
        med_lat_map = self.regions_mask[..., 1]

        axial_names = ["superficial", "deep", "total"]
        sagittal_names = ["medial", "lateral"]

        pd_header = ["Subject", "Location", "Condyle", "Mean", "Std", "Median"]
        pd_list = []

        regions = itertools.product(
            [self._REGION_SUPERFICIAL_KEY, self._REGION_DEEP_KEY, self._TOTAL_AXIAL_KEY],
            [self._MEDIAL_KEY, self._LATERAL_KEY],
        )
        for axial, sagittal in regions:
            if axial == self._TOTAL_AXIAL_KEY:
                axial_map = np.asarray(
                    deep_superficial_map == self._REGION_SUPERFICIAL_KEY, dtype=np.float32
                ) + np.asarray(deep_superficial_map == self._REGION_DEEP_KEY, dtype=np.float32)
                axial_map = np.asarray(axial_map, dtype=bool)
            else:
                axial_map = deep_superficial_map == axial

            sagittal_map = med_lat_map == sagittal

            curr_region_mask = quant_map_volume * axial_map * sagittal_map
            curr_region_mask = curr_region_mask[curr_region_mask != 0]

            # discard all values that are 0
            c_mean = np.nanmean(curr_region_mask)
            c_std = np.nanstd(curr_region_mask)
            c_median = np.nanmedian(curr_region_mask)

            row_info = [
                subject_pid,
                axial_names[axial],
                sagittal_names[sagittal],
                c_mean,
                c_std,
                c_median,
            ]

            pd_list.append(row_info)

        # Generate 2D unrolled matrix
        total, superficial, deep = self.unroll_coronal(quant_map.volume)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [
            {
                "title": "%s superficial" % qv_name,
                "data": superficial,
                "xlabel": "Slice",
                "ylabel": "Angle (binned)",
                "filename": "%s_superficial" % qv_name,
                "raw_data_filename": "%s_superficial.data" % qv_name,
            },
            {
                "title": "%s deep" % qv_name,
                "data": deep,
                "xlabel": "Slice",
                "ylabel": "Angle (binned)",
                "filename": "%s_deep" % qv_name,
                "raw_data_filename": "%s_deep.data" % qv_name,
            },
            {
                "title": "%s total" % qv_name,
                "data": total,
                "xlabel": "Slice",
                "ylabel": "Angle (binned)",
                "filename": "%s_total" % qv_name,
                "raw_data_filename": "%s_total.data" % qv_name,
            },
        ]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask, use_largest_cc: bool = True):
        xp = get_array_module(mask.A)
        if use_largest_cc:
            msk = xp.asarray(largest_cc(mask.A), dtype=xp.uint8)
        else:
            msk = xp.asarray(mask.A, dtype=xp.uint8)
        mask_copy = mask._partial_clone(volume=msk)
        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of patellar cartilage

        Check which quantitative values (T2, T1rho, etc) are defined for
        patellar cartilage and analyze these:

        1. Save 2D total, superficial, and deep visualization maps
        2. Save {'medial', 'lateral'}, {'anterior', 'posterior'},
            {'superior', 'inferior', 'total'} data to excel file

        :param dirpath: base filepath to save data
        """
        q_names = []
        dfs = []

        for quant_val in QuantitativeValueType:
            if quant_val.name not in self.quant_vals.keys():
                continue

            q_names.append(quant_val.name)
            q_val = self.quant_vals[quant_val.name]
            dfs.append(q_val[1])

            q_name_dirpath = io_utils.mkdirs(os.path.join(dirpath, quant_val.name.lower()))
            for q_map_data in q_val[0]:
                filepath = os.path.join(q_name_dirpath, q_map_data["filename"])
                xlabel = ""
                ylabel = ""
                title = q_map_data["title"]
                data_map = q_map_data["data"]

                axs_bounds = self.__get_axis_bounds__(data_map, leave_buffer=True)

                plt.clf()

                upper_bound = BOUNDS[quant_val]
                if preferences.visualization_use_vmax:
                    # Hard bounds - clipping
                    plt.imshow(data_map, cmap="jet", vmin=0.0, vmax=BOUNDS[quant_val])
                else:
                    # Try to use a soft bounds
                    if np.sum(data_map <= upper_bound) == 0:
                        plt.imshow(data_map, cmap="jet", vmin=0.0, vmax=BOUNDS[quant_val])
                    else:
                        warnings.warn(
                            "%s: Pixel value exceeded upper bound (%0.1f). Using normalized scale."
                            % (quant_val.name, upper_bound),
                            stacklevel=2,
                        )
                        plt.imshow(data_map, cmap="jet")

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.ylim(axs_bounds[0])
                plt.gca().invert_yaxis()
                plt.xlim(axs_bounds[1])
                # plt.axis('tight')
                clb = plt.colorbar()
                clb.ax.set_ylabel("(ms)")
                plt.savefig(filepath)

                # Save data
                raw_data_filepath = os.path.join(
                    q_name_dirpath, "raw_data", q_map_data["raw_data_filename"]
                )
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, "data.xlsx"), dfs, q_names)
