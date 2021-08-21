"""Analysis for meniscus.

Attributes:
    BOUNDS (dict): Upper bounds for quantitative values.
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage as sni

from dosma.core.device import get_array_module
from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import T2, QuantitativeValueType
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

__all__ = ["Meniscus"]


class Meniscus(Tissue):
    """Handles analysis and visualization for meniscus.

    This class extends functionality from `Tissue`.

    For visualization, the meniscus is unrolled across the axial plane.
    """

    ID = 2
    STR_ID = "men"
    FULL_NAME = "meniscus"

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    # Coronal Keys
    _ANTERIOR_KEY = 0
    _POSTERIOR_KEY = 1
    _CORONAL_KEYS = [_ANTERIOR_KEY, _POSTERIOR_KEY]

    # Saggital Keys
    _MEDIAL_KEY = 0
    _LATERAL_KEY = 1
    _SAGGITAL_KEYS = [_MEDIAL_KEY, _LATERAL_KEY]

    # Axial Keys
    _SUPERIOR_KEY = 0
    _INFERIOR_KEY = 1
    _TOTAL_AXIAL_KEY = -1

    def __init__(
        self, weights_dir: str = None, medial_to_lateral: bool = None, split_ml_only: bool = False
    ):
        super().__init__(weights_dir=weights_dir, medial_to_lateral=medial_to_lateral)

        self.split_ml_only = split_ml_only
        self.regions_mask = None

    def unroll_axial(self, quant_map: np.ndarray):
        """Unroll meniscus in axial direction.

        Args:
            quant_map (np.ndarray): Map to roll out.

        """
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
        """Split meniscus into subregions.

        Center-of-mass (COM) is used to subdivide into
        anterior/posterior, superior/inferior, and medial/lateral regions.

        Note:
            The anterior/posterior and superior/inferior subdivision may causes issues
            with tilted mensici. This will be addressed in a later release. To avoid
            computing metrics on these regions, set ``self.split_ml_only=True``.
        """
        center_of_mass = sni.measurements.center_of_mass(base_map)  # zero indexed

        com_sup_inf = int(np.ceil(center_of_mass[0]))
        com_ant_post = int(np.ceil(center_of_mass[1]))
        com_med_lat = int(np.ceil(center_of_mass[2]))

        region_mask_sup_inf = np.zeros(base_map.shape)
        region_mask_sup_inf[:com_sup_inf, :, :] = self._SUPERIOR_KEY
        region_mask_sup_inf[com_sup_inf:, :, :] = self._INFERIOR_KEY

        region_mask_ant_post = np.zeros(base_map.shape)
        region_mask_ant_post[:, :com_ant_post, :] = self._ANTERIOR_KEY
        region_mask_ant_post[:, com_ant_post:, :] = self._POSTERIOR_KEY

        region_mask_med_lat = np.zeros(base_map.shape)
        region_mask_med_lat[:, :, :com_med_lat] = (
            self._MEDIAL_KEY if self.medial_to_lateral else self._LATERAL_KEY
        )
        region_mask_med_lat[:, :, com_med_lat:] = (
            self._LATERAL_KEY if self.medial_to_lateral else self._MEDIAL_KEY
        )

        self.regions_mask = np.stack(
            [region_mask_sup_inf, region_mask_ant_post, region_mask_med_lat], axis=-1
        )

    def __calc_quant_vals__(self, quant_map: MedicalVolume, map_type: QuantitativeValueType):
        subject_pid = self.pid

        # Reformats the quantitative map to the appropriate orientation.
        super().__calc_quant_vals__(quant_map, map_type)

        assert (
            self.regions_mask is not None
        ), "region_mask not initialized. Should be initialized when mask is set"

        region_mask = self.regions_mask
        axial_region_mask = self.regions_mask[..., 0]
        coronal_region_mask = self.regions_mask[..., 1]
        sagittal_region_mask = self.regions_mask[..., 2]

        # Combine region mask into categorical mask.
        axial_categories = [
            (self._SUPERIOR_KEY, "superior"),
            (self._INFERIOR_KEY, "inferior"),
            (-1, "total"),
        ]
        coronal_categories = [
            (self._ANTERIOR_KEY, "anterior"),
            (self._POSTERIOR_KEY, "posterior"),
            (-1, "total"),
        ]
        sagittal_categories = [(self._MEDIAL_KEY, "medial"), (self._LATERAL_KEY, "lateral")]
        if self.split_ml_only:
            axial_categories = [x for x in axial_categories if x[0] == -1]
            coronal_categories = [x for x in coronal_categories if x[0] == -1]

        categorical_mask = np.zeros(region_mask.shape[:-1])
        base_mask = self.__mask__.A.astype(np.bool)
        labels = {}
        for idx, (
            (axial, axial_name),
            (coronal, coronal_name),
            (sagittal, sagittal_name),
        ) in enumerate(
            itertools.product(axial_categories, coronal_categories, sagittal_categories)
        ):
            label = idx + 1
            axial_map = np.asarray([True]) if axial == -1 else axial_region_mask == axial
            coronal_map = np.asarray([True]) if coronal == -1 else coronal_region_mask == coronal
            sagittal_map = sagittal_region_mask == sagittal
            categorical_mask[base_mask & axial_map & coronal_map & sagittal_map] = label
            labels[label] = f"{axial_name}-{coronal_name}-{sagittal_name}"

        # TODO: Change this to be any arbitrary quantitative value type.
        # Note, it does not matter what we wrap it in because the underlying operations
        # are not specific to the value type.
        t2 = T2(quant_map)
        categorical_mask = MedicalVolume(categorical_mask, affine=quant_map.affine)
        df = t2.to_metrics(categorical_mask, labels=labels, bounds=(0, np.inf), closed="neither")
        df.insert(0, "Subject", subject_pid)

        total, superior, inferior = self.unroll_axial(quant_map.volume)
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

    def __calc_quant_vals_old__(self, quant_map, map_type):
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
        sagittal_names = ["anterior", "posterior"]

        pd_header = ["Subject", "Location", "Side", "Region", "Mean", "Std", "Median"]
        pd_list = []

        for axial in [self._SUPERIOR_KEY, self._INFERIOR_KEY, self._TOTAL_AXIAL_KEY]:
            if axial == self._TOTAL_AXIAL_KEY:
                axial_map = np.asarray(
                    axial_region_mask == self._SUPERIOR_KEY, dtype=np.float32
                ) + np.asarray(axial_region_mask == self._INFERIOR_KEY, dtype=np.float32)
                axial_map = np.asarray(axial_map, dtype=np.bool)
            else:
                axial_map = axial_region_mask == axial

            for coronal in [self._MEDIAL_KEY, self._LATERAL_KEY]:
                for sagittal in [self._ANTERIOR_KEY, self._POSTERIOR_KEY]:
                    curr_region_mask = (
                        quant_map_volume
                        * (coronal_region_mask == coronal)
                        * (sagittal_region_mask == sagittal)
                        * axial_map
                    )
                    curr_region_mask[curr_region_mask == 0] = np.nan
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

    def set_mask(self, mask: MedicalVolume, use_largest_ccs: bool = False, ml_only: bool = False):
        xp = get_array_module(mask.A)
        if use_largest_ccs:
            msk = xp.asarray(largest_cc(mask.A, num=2), dtype=xp.uint8)
        else:
            msk = xp.asarray(mask.A, dtype=xp.uint8)
        mask_copy = mask._partial_clone(volume=msk)
        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of meniscus

        Check which quantitative values (T2, T1rho, etc) are defined for meniscus and analyze these
            1. Save 2D total, superficial, and deep visualization maps
            2. Save {'medial', 'lateral'}, {'anterior', 'posterior'},
                {'superior', 'inferior', 'total'} data to excel file.

        Args:
            dirpath (str): Directory path to tissue data.
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
                            % (quant_val.name, upper_bound)
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
