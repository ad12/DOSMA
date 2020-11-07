"""Analysis for patellar cartilage.

Attributes:
    BOUNDS (dict): Upper bounds for quantitative values.
"""
import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as sni

from dosma.tissues.tissue import Tissue

from dosma.defaults import preferences
from dosma.utils import io_utils
from dosma.quant_vals import QuantitativeValueType

# milliseconds
BOUNDS = {QuantitativeValueType.T2: 60.0,
          QuantitativeValueType.T1_RHO: 100.0,
          QuantitativeValueType.T2_STAR: 50.0}

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

    def __init__(self,  weights_dir: str = None, medial_to_lateral: bool = None):
        super().__init__(weights_dir=weights_dir, medial_to_lateral=medial_to_lateral)

        self.regions_mask = None

    def unroll_coronal(self, quant_map: np.ndarray):
        """Unroll patellar cartilage in the coronal direction.

        Because patellar cartilage is flat, "unrolling" is projecting the patellar cartilage onto the coronal axis.

        Args:
            quant_map (np.ndarray):
        """
        mask = self.__mask__.volume

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"
        region_mask_deep_superficial = self.regions_mask[..., 0]

        superficial = (region_mask_deep_superficial == self._REGION_SUPERFICIAL_KEY) * mask * quant_map
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
        """Split patellar cartilage into deep/superficial regions"""
        if np.sum(base_map) == 0:
            warnings.warn('No mask for `%s` was found.' % self.FULL_NAME)

        self.regions_mask = self.__2d_split_regions(base_map)

    def __2d_split_regions(self, base_map):
        """Split patellar cartilage into deep/superficial regions per sagittal slice

        Left = Superficial, Right = deep
        For patellar cartilage, the superficial-->deep transition happens in the anterior-->posterior direction

        TODO (arjundd): refactor to make region map a Medical Volume
        """
        region_mask_sup_deep = np.zeros(base_map.shape)

        for s in range(base_map.shape[-1]):
            c_slice = base_map[..., s]
            if np.sum(c_slice) == 0:
                ds_split = 0
            else:
                center_of_mass = sni.measurements.center_of_mass(c_slice)
                ds_split = int(center_of_mass[1])
            com_deep_superficial = ds_split
            region_mask_sup_deep[:, :com_deep_superficial, s] = self._REGION_SUPERFICIAL_KEY
            region_mask_sup_deep[:, com_deep_superficial:, s] = self._REGION_DEEP_KEY

        return region_mask_sup_deep[..., np.newaxis]

    def __calc_quant_vals__(self, quant_map, map_type):
        subject_pid = self.pid

        super().__calc_quant_vals__(quant_map, map_type)

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        quant_map_volume = quant_map.volume
        mask = self.__mask__.volume
        quant_map_volume = mask * quant_map_volume

        deep_superficial_map = self.regions_mask[..., 0]

        axial_names = ['superficial', 'deep', 'total']

        pd_header = ['Subject', 'Location', 'Mean', 'Std', 'Median']
        pd_list = []

        for axial in [self._REGION_SUPERFICIAL_KEY, self._REGION_DEEP_KEY, self._TOTAL_AXIAL_KEY]:
            if axial == self._TOTAL_AXIAL_KEY:
                axial_map = np.asarray(deep_superficial_map == self._REGION_SUPERFICIAL_KEY, dtype=np.float32) + \
                            np.asarray(deep_superficial_map == self._REGION_DEEP_KEY, dtype=np.float32)
                axial_map = np.asarray(axial_map, dtype=np.bool)
            else:
                axial_map = deep_superficial_map == axial

            curr_region_mask = quant_map_volume * axial_map
            curr_region_mask[curr_region_mask == 0] = np.nan

            # discard all values that are 0
            c_mean = np.nanmean(curr_region_mask)
            c_std = np.nanstd(curr_region_mask)
            c_median = np.nanmedian(curr_region_mask)

            row_info = [subject_pid, axial_names[axial], c_mean, c_std, c_median]

            pd_list.append(row_info)

        # Generate 2D unrolled matrix
        total, superficial, deep = self.unroll_coronal(quant_map.volume)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [{'title': '%s superficial' % qv_name, 'data': superficial, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_superficial' % qv_name, 'raw_data_filename': '%s_superficial.data' % qv_name},
                {'title': '%s deep' % qv_name, 'data': deep, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_deep' % qv_name, 'raw_data_filename': '%s_deep.data' % qv_name},
                {'title': '%s total' % qv_name, 'data': total, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_total' % qv_name, 'raw_data_filename': '%s_total.data' % qv_name}]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask):
        mask_copy = deepcopy(mask)
        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of patellar cartilage

        Check which quantitative values (T2, T1rho, etc) are defined for patellar cartilage and analyze these
        1. Save 2D total, superficial, and deep visualization maps
        2. Save {'medial', 'lateral'}, {'anterior', 'posterior'}, {'superior', 'inferior', 'total'} data to excel
               file

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
                filepath = os.path.join(q_name_dirpath, q_map_data['filename'])
                xlabel = ''
                ylabel = ''
                title = q_map_data['title']
                data_map = q_map_data['data']

                axs_bounds = self.__get_axis_bounds__(data_map, leave_buffer=True)

                plt.clf()

                upper_bound = BOUNDS[quant_val]
                if preferences.visualization_use_vmax:
                    # Hard bounds - clipping
                    plt.imshow(data_map, cmap='jet', vmin=0.0, vmax=BOUNDS[quant_val])
                else:
                    # Try to use a soft bounds
                    if np.sum(data_map <= upper_bound) == 0:
                        plt.imshow(data_map, cmap='jet', vmin=0.0, vmax=BOUNDS[quant_val])
                    else:
                        warnings.warn('%s: Pixel value exceeded upper bound (%0.1f). Using normalized scale.'
                                      % (quant_val.name, upper_bound))
                        plt.imshow(data_map, cmap='jet')

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.ylim(axs_bounds[0])
                plt.gca().invert_yaxis()
                plt.xlim(axs_bounds[1])
                #plt.axis('tight')
                clb = plt.colorbar()
                clb.ax.set_ylabel('(ms)')
                plt.savefig(filepath)

                # Save data
                raw_data_filepath = os.path.join(q_name_dirpath, 'raw_data', q_map_data['raw_data_filename'])
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)
