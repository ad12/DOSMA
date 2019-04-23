import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as sni

import defaults
from data_io.med_volume import MedicalVolume
from tissues.tissue import Tissue
from utils import io_utils
from utils.quant_vals import QuantitativeValues
from data_io.fig_format import savefig


# milliseconds
BOUNDS = {QuantitativeValues.T2: 60.0,
          QuantitativeValues.T1_RHO: 100.0,
          QuantitativeValues.T2_STAR: 50.0}


class Meniscus(Tissue):
    """Handles analysis for meniscus"""
    ID = 2
    STR_ID = 'men'
    FULL_NAME = 'meniscus'

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    # Coronal Keys
    ANTERIOR_KEY = 0
    POSTERIOR_KEY = 1
    CORONAL_KEYS = [ANTERIOR_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 0
    LATERAL_KEY = 1
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    # Axial Keys
    SUPERIOR_KEY = 0
    INFERIOR_KEY = 1
    TOTAL_AXIAL_KEY = -1

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        """
        :param weights_dir: Directory to weights files
        :param medial_to_lateral: True or False, if false, then lateral to medial
        """
        super().__init__(weights_dir=weights_dir)

        self.regions_mask = None
        self.medial_to_lateral = medial_to_lateral

    def unroll_axial(self, quant_map):
        mask = self.__mask__.volume

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"
        region_mask_sup_inf = self.regions_mask[..., 0]

        superior = (region_mask_sup_inf == self.SUPERIOR_KEY) * mask * quant_map
        superior[superior == 0] = np.nan
        superior = np.nanmean(superior, axis=0)

        inferior = (region_mask_sup_inf == self.INFERIOR_KEY) * mask * quant_map
        inferior[inferior == 0] = np.nan
        inferior = np.nanmean(inferior, axis=0)

        total = mask * quant_map
        total[total == 0] = np.nan
        total = np.nanmean(total, axis=0)

        return total, superior, inferior

    def split_regions(self, base_map):
        center_of_mass = sni.measurements.center_of_mass(base_map)  # zero indexed

        com_sup_inf = int(np.ceil(center_of_mass[0]))
        com_ant_post = int(np.ceil(center_of_mass[1]))
        com_med_lat = int(np.ceil(center_of_mass[2]))

        region_mask_sup_inf = np.zeros(base_map.shape)
        region_mask_sup_inf[:com_sup_inf, :, :] = self.SUPERIOR_KEY
        region_mask_sup_inf[com_sup_inf:, :, :] = self.INFERIOR_KEY

        region_mask_ant_post = np.zeros(base_map.shape)
        region_mask_ant_post[:, :com_ant_post, :] = self.ANTERIOR_KEY
        region_mask_ant_post[:, com_ant_post:, :] = self.POSTERIOR_KEY

        region_mask_med_lat = np.zeros(base_map.shape)
        region_mask_med_lat[:, :, :com_med_lat] = self.MEDIAL_KEY if self.medial_to_lateral else self.LATERAL_KEY
        region_mask_med_lat[:, :, com_med_lat:] = self.LATERAL_KEY if self.medial_to_lateral else self.MEDIAL_KEY

        self.regions_mask = np.stack([region_mask_sup_inf, region_mask_ant_post, region_mask_med_lat], axis=-1)

    def __calc_quant_vals__(self, quant_map, map_type):
        subject_pid = self.pid

        super().__calc_quant_vals__(quant_map, map_type)

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        quant_map_volume = quant_map.volume
        mask = self.__mask__.volume

        quant_map_volume = mask * quant_map_volume

        axial_region_mask = self.regions_mask[..., 0]
        sagittal_region_mask = self.regions_mask[..., 1]
        coronal_region_mask = self.regions_mask[..., 2]

        axial_names = ['superior', 'inferior', 'total']
        coronal_names = ['medial', 'lateral']
        sagittal_names = ['anterior', 'posterior']

        pd_header = ['Subject', 'Location', 'Side', 'Region', 'Mean', 'Std', 'Median']
        pd_list = []

        for axial in [self.SUPERIOR_KEY, self.INFERIOR_KEY, self.TOTAL_AXIAL_KEY]:
            if axial == self.TOTAL_AXIAL_KEY:
                axial_map = np.asarray(axial_region_mask == self.SUPERIOR_KEY, dtype=np.float32) + \
                            np.asarray(axial_region_mask == self.INFERIOR_KEY, dtype=np.float32)
                axial_map = np.asarray(axial_map, dtype=np.bool)
            else:
                axial_map = axial_region_mask == axial

            for coronal in [self.MEDIAL_KEY, self.LATERAL_KEY]:
                for sagittal in [self.ANTERIOR_KEY, self.POSTERIOR_KEY]:
                    curr_region_mask = quant_map_volume * (coronal_region_mask == coronal) * (
                            sagittal_region_mask == sagittal) * axial_map
                    curr_region_mask[curr_region_mask == 0] = np.nan
                    # discard all values that are 0
                    c_mean = np.nanmean(curr_region_mask)
                    c_std = np.nanstd(curr_region_mask)
                    c_median = np.nanmedian(curr_region_mask)

                    row_info = [subject_pid, axial_names[axial], coronal_names[coronal], sagittal_names[sagittal],
                                c_mean, c_std, c_median]

                    pd_list.append(row_info)

        # Generate 2D unrolled matrix
        total, superior, inferior = self.unroll_axial(quant_map.volume)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [{'title': '%s superior' % qv_name, 'data': superior, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_superior.png' % qv_name, 'raw_data_filename': '%s_superior.data' % qv_name},
                {'title': '%s inferior' % qv_name, 'data': inferior, 'xlabel': 'Slice',
                 'ylabel': 'Angle (binned)', 'filename': '%s_inferior.png' % qv_name,
                 'raw_data_filename': '%s_inferior.data' % qv_name},
                {'title': '%s total' % qv_name, 'data': total, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_total.png' % qv_name,
                 'raw_data_filename': '%s_total.data' % qv_name}]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask: MedicalVolume):
        mask_copy = deepcopy(mask)

        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of meniscus

        Check which quantitative values (T2, T1rho, etc) are defined for meniscus and analyze these
        1. Save 2D total, superficial, and deep visualization maps
        2. Save {'medial', 'lateral'}, {'anterior', 'posterior'}, {'superior', 'inferior', 'total'} data to excel
               file

        :param dirpath: base filepath to save data
        """
        q_names = []
        dfs = []

        for quant_val in QuantitativeValues:
            if quant_val.name not in self.quant_vals.keys():
                continue

            q_names.append(quant_val.name)
            q_val = self.quant_vals[quant_val.name]
            dfs.append(q_val[1])

            q_name_dirpath = io_utils.check_dir(os.path.join(dirpath, quant_val.name.lower()))
            for q_map_data in q_val[0]:
                filepath = os.path.join(q_name_dirpath, q_map_data['filename'])
                xlabel = 'Slice'
                ylabel = ''
                title = q_map_data['title']
                data_map = q_map_data['data']

                plt.clf()

                upper_bound = BOUNDS[quant_val]
                is_picture_written = False

                if defaults.VISUALIZATION_HARD_BOUNDS:
                    plt.imshow(data_map, cmap='jet', vmin=0.0, vmax=BOUNDS[quant_val])
                    is_picture_written = True

                if defaults.VISUALIZATION_SOFT_BOUNDS and not is_picture_written:
                    if np.sum(data_map <= upper_bound) == 0:
                        plt.imshow(data_map, cmap='jet', vmin=0.0, vmax=BOUNDS[quant_val])
                        is_picture_written = True
                    else:
                        warnings.warn('%s: Pixel value exceeded upper bound (%0.1f). Using normalized scale.'
                                      % (quant_val.name, upper_bound))

                if not is_picture_written:
                    plt.imshow(data_map, cmap='jet')

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                clb = plt.colorbar()
                clb.ax.set_title('(ms)')
                plt.axis('tight')

                savefig(filepath, dpi=defaults.DEFAULT_DPI)

                # Save data
                raw_data_filepath = os.path.join(q_name_dirpath, 'raw_data', q_map_data['raw_data_filename'])
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)
