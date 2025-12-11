"""Analysis for tibial cartilage.

Attributes:
    BOUNDS (dict): Upper bounds for quantitative values.
"""
import os
import warnings

import numpy as np
import pandas as pd

from dosma.core.device import get_array_module
from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import QuantitativeValueType
from dosma.defaults import preferences
from dosma.tissues.tissue import Tissue, largest_cc
from dosma.utils import geometry_utils, io_utils

import matplotlib.pyplot as plt

# milliseconds
BOUNDS = {
    QuantitativeValueType.T2: 60.0,
    QuantitativeValueType.T1_RHO: 100.0,
    QuantitativeValueType.T2_STAR: 50.0,
}

__all__ = ["TibialCartilage"]


class TibialCartilage(Tissue):
    """Handles analysis and visualization for tibial cartilage."""

    ID = 4
    STR_ID = "tc"
    FULL_NAME = "tibial cartilage"

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    # Coronal Keys
    _ANTERIOR_KEY = 0
    _POSTERIOR_KEY = 1
    _CENTRAL_KEY = 2
    _CORONAL_KEYS = [_ANTERIOR_KEY, _CENTRAL_KEY, _POSTERIOR_KEY]

    # Saggital Keys
    _MEDIAL_KEY = 0
    _LATERAL_KEY = 1
    _SAGITTAL_KEYS = [_MEDIAL_KEY, _LATERAL_KEY]

    # Axial Keys
    _SUPERIOR_KEY = 0
    _INFERIOR_KEY = 1
    _TOTAL_AXIAL_KEY = -1

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        super().__init__(weights_dir=weights_dir, medial_to_lateral=medial_to_lateral)

        self.regions_mask = None

    def unroll_axial(self, quant_map):
        mask = self.__mask__.volume

        assert (
            self.regions_mask is not None
        ), "region_mask not initialized. Should be initialized when mask is set"
        region_mask_sup_inf = self.regions_mask[..., 0]

        superior = (region_mask_sup_inf == self._SUPERIOR_KEY) * mask * quant_map
        superior[superior == 0] = np.nan
        superior = np.nanmean(superior, axis=0)

        inferior = (region_mask_sup_inf == self._INFERIOR_KEY) * mask * quant_map
        inferior[inferior == 0] = np.nan
        inferior = np.nanmean(inferior, axis=0)

        total = mask * quant_map
        total[total == 0] = np.nan
        total = np.nanmean(total, axis=0)

        return total, superior, inferior

    def split_regions(self, base_map):
        """Generate subregions for tibial cartilage.

        Tibial cartilage is split into subregions along the 3 major axes:
        superior/inferior (S/I), anterior/posterior (A/P), medial/lateral (M/L).
        M/L plateaus are computed with respect to the center of mass (COM)
        in the sagittal direction. A/C/P divisions are computed independently for each
        plateau using thirds of the distance between the minimum/maximum pixel in the A/P direction.
        S/I divisions are computed based on local COM for each 1D column with
        tissues in the S/I direction.

        The output is stored in `self.regions_mask`.

        Note:
            Previously, COM was used to split A/P subregions. However, COM is not as robust
            to split the data into 3 subregions. We use the method described in the reference
            below: A/C/P are split by thirds of the distance between the minimum/maximum pixel
            in the A/P direction. Note the minimum/maximum pixel method may not be a robust if
            there are a sufficient number of erroneous pixels. We do not sufficiently correct
            for erroneous pixels.

        Args:
            base_map (ndarray): Binary 3D mask with orientation (SI, AP, ML/LM).
                If `self.medial_to_lateral`, last dimension should be ML.

        References:
            Black, MS et al. Detecting Early Changes in ACL-Reconstructed Knees:
            Cluster Analysis of T2 Relaxation Times from 3 Months to 18 Months Post-Surgery.
            28th Annual Meeting of ISMRM, Sydney, Australia 2020.
        """
        xp = get_array_module(base_map)
        center_of_mass = geometry_utils.center_of_mass(base_map)  # zero indexed
        com_med_lat = int(xp.ceil(center_of_mass[2]))

        # M/L
        region_mask_med_lat = xp.zeros(base_map.shape)
        region_mask_med_lat[:, :, :com_med_lat] = (
            self._MEDIAL_KEY if self.medial_to_lateral else self._LATERAL_KEY
        )
        region_mask_med_lat[:, :, com_med_lat:] = (
            self._LATERAL_KEY if self.medial_to_lateral else self._MEDIAL_KEY
        )

        # S/I
        locs = base_map.sum(axis=0).nonzero()
        voxels = base_map[:, locs[0], locs[1]]
        com_sup_inf = xp.asarray(
            [
                int(xp.ceil(geometry_utils.center_of_mass(voxels[:, i])[0]))
                for i in range(voxels.shape[1])
            ]
        )
        region_mask_sup_inf = xp.full(base_map.shape, self._INFERIOR_KEY)
        for i in range(len(com_sup_inf)):
            region_mask_sup_inf[
                : com_sup_inf[i].item(), locs[0][i].item(), locs[1][i].item()
            ] = self._SUPERIOR_KEY

        # A/C/P
        region_mask_ant_post = xp.zeros(base_map.shape)
        for plateau in [slice(0, com_med_lat), slice(com_med_lat, None)]:
            cum_ap = xp.nonzero(base_map[..., plateau].sum(axis=(0, 2)))[0]
            min_ap = xp.min(cum_ap)
            ap_range = xp.max(cum_ap) - min_ap
            thresh1, thresh2 = (
                int(xp.ceil(min_ap + 1 / 3 * ap_range)),
                int(xp.ceil(min_ap + 2 / 3 * ap_range)),
            )
            region_mask_ant_post[:, :thresh1, plateau] = self._ANTERIOR_KEY
            region_mask_ant_post[:, thresh1:thresh2, plateau] = self._CENTRAL_KEY
            region_mask_ant_post[:, thresh2:, plateau] = self._POSTERIOR_KEY

        # # A/P
        # region_mask_ant_post = np.zeros(base_map.shape)
        # for plateau in [slice(0, com_med_lat), slice(com_med_lat, None)]:
        #     com = sni.measurements.center_of_mass(base_map[..., plateau])
        #     com_ant_post = int(np.ceil(com[1]))
        #     region_mask_ant_post[:, :com_ant_post, plateau] = self._ANTERIOR_KEY
        #     region_mask_ant_post[:, com_ant_post:, plateau] = self._POSTERIOR_KEY

        self.regions_mask = xp.stack(
            [region_mask_sup_inf, region_mask_ant_post, region_mask_med_lat], axis=-1
        )

    def __calc_quant_vals__(self, quant_map, map_type):
        subject_pid = self.pid

        super().__calc_quant_vals__(quant_map, map_type)

        assert (
            self.regions_mask is not None
        ), "region_mask not initialized. Should be initialized when mask is set"

        quant_map_volume = quant_map.volume
        mask = self.__mask__.volume

        quant_map_volume = mask * quant_map_volume

        axial_region_mask = self.regions_mask[..., 0]
        sagittal_region_mask = self.regions_mask[..., 1]
        coronal_region_mask = self.regions_mask[..., 2]

        axial_names = ["superior", "inferior", "total"]
        coronal_names = ["medial", "lateral"]
        sagittal_names = ["anterior", "posterior", "central"]

        pd_header = ["Subject", "Location", "Side", "Region", "Mean", "Std", "Median"]
        pd_list = []

        for axial in [self._SUPERIOR_KEY, self._INFERIOR_KEY, self._TOTAL_AXIAL_KEY]:
            if axial == self._TOTAL_AXIAL_KEY:
                axial_map = (axial_region_mask == self._SUPERIOR_KEY).astype(np.float32) + (
                    axial_region_mask == self._INFERIOR_KEY
                ).astype(np.float32)
                axial_map = axial_map.astype(bool)
            else:
                axial_map = axial_region_mask == axial

            for coronal in [self._MEDIAL_KEY, self._LATERAL_KEY]:
                for sagittal in [self._ANTERIOR_KEY, self._POSTERIOR_KEY, self._CENTRAL_KEY]:
                    curr_region_mask = (
                        quant_map_volume
                        * (coronal_region_mask == coronal)
                        * (sagittal_region_mask == sagittal)
                        * axial_map
                    )
                    curr_region_mask = curr_region_mask[curr_region_mask != 0]
                    # discard all values that are 0
                    c_mean = np.nanmean(curr_region_mask)
                    c_std = np.nanstd(curr_region_mask)
                    c_median = np.nanmedian(curr_region_mask)

                    row_info = [
                        subject_pid,
                        axial_names[axial],
                        coronal_names[coronal],
                        sagittal_names[sagittal],
                        c_mean,
                        c_std,
                        c_median,
                    ]

                    pd_list.append(row_info)

        # Generate 2D unrolled matrix
        total, superior, inferior = self.unroll_axial(quant_map.volume)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [
            {
                "title": "%s superior" % qv_name,
                "data": superior,
                "xlabel": "Slice",
                "ylabel": "Angle (binned)",
                "filename": "%s_superior" % qv_name,
                "raw_data_filename": "%s_superior.data" % qv_name,
            },
            {
                "title": "%s inferior" % qv_name,
                "data": inferior,
                "xlabel": "Slice",
                "ylabel": "Angle (binned)",
                "filename": "%s_inferior" % qv_name,
                "raw_data_filename": "%s_inferior.data" % qv_name,
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

    def set_mask(self, mask: MedicalVolume, use_largest_ccs=False):
        xp = get_array_module(mask.A)
        if use_largest_ccs:
            msk = xp.asarray(largest_cc(mask.A, num=2), dtype=xp.uint8)
        else:
            msk = xp.asarray(mask.A, dtype=xp.uint8)
        mask_copy = mask._partial_clone(volume=msk)
        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of tibial cartilage
        Check which quantitative values (T2, T1rho, etc) are defined for meniscus and analyze these
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
                xlabel = "Slice"
                ylabel = ""
                title = q_map_data["title"]
                data_map = q_map_data["data"]

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
                clb = plt.colorbar()
                clb.ax.set_title("(ms)")
                plt.axis("tight")

                plt.savefig(filepath)

                # Save data
                raw_data_filepath = os.path.join(
                    q_name_dirpath, "raw_data", q_map_data["raw_data_filename"]
                )
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, "data.xlsx"), dfs, q_names)
