import os
import warnings

import matplotlib.pyplot as plt
import nipy.labs.mask as nlm
import numpy as np
import pandas as pd
import scipy.ndimage as sni
import time

import defaults
from med_objects.med_volume import MedicalVolume
from tissues.tissue import Tissue
from utils import io_utils, img_utils
from utils.geometry_utils import circle_fit, cart2pol
from utils.quant_vals import QuantitativeValues

# milliseconds
BOUNDS = {QuantitativeValues.T2: 60.0,
          QuantitativeValues.T1_RHO: 100.0,
          QuantitativeValues.T2_STAR: 50.0}

THICKNESS_DIVISOR = 0.5  # split between superficial/deep cartilage
NB_BINS = 72  # number of bins
DTHETA = 360 / NB_BINS  # theta intervals for bins (in degrees)

# Theta range: [-270, 90)
THETA_MIN = -270
THETA_MAX = 90


class FemoralCartilage(Tissue):
    """Handles analysis for femoral cartilage"""
    ID = 1
    STR_ID = 'fc'
    FULL_NAME = 'femoral cartilage'

    # Expected quantitative values
    T1_EXPECTED = 1200  # milliseconds

    # Coronal Keys
    ANTERIOR_KEY = 0
    CENTRAL_KEY = 1
    POSTERIOR_KEY = 2
    CORONAL_KEYS = [ANTERIOR_KEY, CENTRAL_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 0
    LATERAL_KEY = 1
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    # Axial Keys
    DEEP_KEY = 0
    SUPERFICIAL_KEY = 1
    TOTAL_AXIAL_KEY = 2

    ML_BOUNDARY = None
    ACP_BOUNDARY = None

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        """
        :param weights_dir: Directory to weights files
        :param medial_to_lateral: True or False, if false, then lateral to medial
        """
        super().__init__(weights_dir=weights_dir)

        self.regions_mask = None
        self.theta_bins = None

        self.medial_to_lateral = medial_to_lateral

    def split_regions(self, base_map):
        mask = self.__mask__.volume

        height, width, num_slices = mask.shape

        # STEP 1: PROJECTING AND CYLINDRICAL FIT
        segmented_T2maps_projected = np.max(mask, 2)  # Project segmented T2maps on sagittal axis

        non_zero_element = np.nonzero(segmented_T2maps_projected)

        xc_fit, yc_fit, R_fit = circle_fit(non_zero_element[1],
                                           non_zero_element[0])  # fit a circle to projected cartilage tissue

        # STEP 2: SLICE BY SLICE BINNING
        yv, xv = np.meshgrid(range(height), range(width), indexing='ij')

        rho, theta = cart2pol(xv - xc_fit, yc_fit - yv)
        theta = (theta >= 90) * (theta-360) + (theta < 90)*theta  # range: [-270, 90)

        assert (np.min(theta) >= THETA_MIN) and (np.max(theta) < THETA_MAX), \
            "Expected Theta range is [%d, %d) degrees. Recieved min: %d max: %d)." % (THETA_MIN, THETA_MAX,
                                                                                      np.min(theta), np.max(theta))

        theta_bins = np.floor((theta - THETA_MIN) / DTHETA)

        # STEP 3: COMPUTE THRESHOLD RADII
        rhos_threshold_volume = np.zeros(mask.shape)
        for curr_slice in range(num_slices):
            mask_slice = mask[..., curr_slice]

            for curr_bin in range(NB_BINS):
                rhos_valid = rho[np.logical_and(mask_slice > 0, theta_bins == curr_bin)]
                if len(rhos_valid) == 0:
                    continue

                rho_min = np.min(rhos_valid)
                rho_max = np.max(rhos_valid)

                rho_threshold = THICKNESS_DIVISOR * (rho_max - rho_min) + rho_min
                rhos_threshold_volume[theta_bins == curr_bin, curr_slice] = rho_threshold

        # anterior/central/posterior division
        # Central region occupies middle 30 degrees, anterior on left, posterior on right
        acp_map = np.zeros(theta.shape)
        acp_map[theta < -105] = self.ANTERIOR_KEY
        acp_map[np.logical_and((theta >= -105), (theta < -75))] = self.CENTRAL_KEY
        acp_map[theta >= -75] = self.POSTERIOR_KEY
        acp_volume = np.stack([acp_map]*num_slices, axis=-1)

        # medial/lateral division
        # take into account scanning direction
        center_of_mass = sni.measurements.center_of_mass(mask)
        com_slicewise = center_of_mass[-1]
        ml_volume = np.zeros(mask.shape)

        if self.medial_to_lateral:
            ml_volume[..., :int(np.ceil(com_slicewise))] = self.MEDIAL_KEY
            ml_volume[..., int(np.ceil(com_slicewise)):] = self.LATERAL_KEY
        else:
            ml_volume[..., :int(np.ceil(com_slicewise))] = self.LATERAL_KEY
            ml_volume[..., int(np.ceil(com_slicewise)):] = self.MEDIAL_KEY

        # deep/superficial division
        rho_volume = np.stack([rho]*num_slices, axis=-1)
        ds_volume = np.zeros(mask.shape)
        ds_volume[rho_volume < rhos_threshold_volume] = self.DEEP_KEY
        ds_volume[rho_volume >= rhos_threshold_volume] = self.SUPERFICIAL_KEY
        ds_volume = ds_volume

        self.regions_mask = np.stack([ds_volume, acp_volume, ml_volume], axis=-1)
        self.theta_bins = theta_bins
        self.ML_BOUNDARY = int(np.ceil(com_slicewise))
        self.ACP_BOUNDARY = [int(np.floor((-105 - THETA_MIN) / DTHETA)), int(np.floor((-75 - THETA_MIN) / DTHETA))]

    def unroll(self, qv_map):
        """Unroll femoral cartilage 3D quantitative value (qv) maps to 2D for visualiation

        The function multiplies a 3D segmentation mask to a 3D qv map --> 3D femoral cartilage qv (fc_qv) map
        It then fits a circle to the collapsed sagittal projection of the fc_qv map
        Each slice is binned into bins of 5 degree sizes

        The unrolled map is then divided into deep and superficial cartilage

        :param qv_map: 3D numpy array (slices last) of sagittal knee describing quantitative parameter values
        :return: tuple in (row, column) format:
                    1. 2D Total unrolled cartilage (slices, degrees) - average of superficial and deep layers
                    2. Superficial unrolled cartilage (slices, degrees) - superficial layer
                    3. Deep unrolled cartilage (slices, degrees) - deep layer
        """
        mask = self.__mask__.volume

        if qv_map.shape != mask.shape:
            raise ValueError('t2_map and mask must have same shape')

        if len(qv_map.shape) != 3:
            raise ValueError('t2_map and mask must be 3D')

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        num_slices = qv_map.shape[-1]

        qv_map = np.nan_to_num(qv_map)
        qv_map = np.multiply(mask, qv_map)  # apply binary mask
        qv_map[qv_map == 0] = np.nan  # wherever qv_map is 0, either no cartilage or qv=0 ms, which is impractical

        theta_bins = self.theta_bins  # binning with theta
        ds_volume = self.regions_mask[..., 0]  # deep/superficial split

        Unrolled_Cartilage = np.zeros([NB_BINS, num_slices])
        Sup_layer = np.zeros([NB_BINS, num_slices])
        Deep_layer = np.zeros([NB_BINS, num_slices])

        for curr_slice in range(num_slices):
            qv_slice = qv_map[..., curr_slice]
            ds_slice = ds_volume[..., curr_slice]
            # if slice is all NaNs, then don't analyze
            if np.sum(np.isnan(qv_slice)) == qv_slice.shape[0] * qv_slice.shape[1]:
                continue

            for curr_bin in range(NB_BINS):
                qv_bin = qv_slice[theta_bins == curr_bin]
                if np.sum(np.isnan(qv_bin)) == len(qv_bin):
                    continue

                Unrolled_Cartilage[curr_bin, curr_slice] = np.nanmean(qv_bin)

                qv_superficial = qv_slice[np.logical_and(theta_bins == curr_bin, ds_slice == self.SUPERFICIAL_KEY)]
                qv_deep = qv_slice[np.logical_and(theta_bins == curr_bin, ds_slice == self.DEEP_KEY)]
                # assert len(qv_superficial) > 1, "must have at least 1 superficial pixel"
                # assert len(qv_deep) >

                if len(qv_superficial) <= 1 and np.sum(np.isnan(qv_superficial)) == len(qv_superficial):
                    import pdb; pdb.set_trace()

                if len(qv_deep) <= 1 and np.sum(np.isnan(qv_deep)) == len(qv_deep):
                    import pdb; pdb.set_trace()

                Sup_layer[curr_bin, curr_slice] = np.nanmean(qv_superficial)
                Deep_layer[curr_bin, curr_slice] = np.nanmean(qv_deep)

                assert np.sum(np.isnan(qv_deep)) != len(qv_deep) or np.sum(np.isnan(qv_superficial)) != len(qv_superficial)

        Unrolled_Cartilage[Unrolled_Cartilage == 0] = np.nan
        Sup_layer[Sup_layer == 0] = np.nan
        Deep_layer[Deep_layer == 0] = np.nan

        return Unrolled_Cartilage, Sup_layer, Deep_layer

    def __calc_quant_vals__(self, quant_map, map_type):
        """
        Calculate quantitative values per region and 2D visualizations

        1. Save 2D figure (deep, superficial, total) information to use with matplotlib
                (title, data, xlabel, ylabel, filename)
        2. Save 2D dataframes in format:
                [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
                 ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
                 ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]

                 D=deep, S=superficial, T=total,
                 M=medial, L=lateral,
                 A=anterior, C=central, P=posterior

        :param quant_map: The 3D volume of quantitative values (np.nan for all pixels that are not accurate)
        :param map_type: A QuantitativeValue instance
        """

        super().__calc_quant_vals__(quant_map, map_type)

        if self.__mask__ is None:
            raise ValueError('Please initialize mask')

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        total, superficial, deep = self.unroll(quant_map.volume)

        assert total.shape == deep.shape
        assert deep.shape == superficial.shape

        axial_region_mask = self.regions_mask[..., 0]
        sagittal_region_mask = self.regions_mask[..., 1]
        coronal_region_mask = self.regions_mask[..., 2]
        mask = self.__mask__.volume

        subject_pid = self.pid
        pd_header = ['Subject', 'Location', 'Side', 'Region', 'Mean', 'Std', 'Median']
        pd_list = []

        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        # tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
        #                  ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
        #                  ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]
        # tissue_values = []
        axial_data = [deep, superficial, total]

        axial_names = ['deep', 'superficial', 'total']
        coronal_names = ['medial', 'lateral']
        sagittal_names = ['anterior', 'central', 'posterior']

        for axial in [self.DEEP_KEY, self.SUPERFICIAL_KEY, self.TOTAL_AXIAL_KEY]:
            if axial == self.TOTAL_AXIAL_KEY:
                axial_map = np.logical_or(np.asarray(axial_region_mask == self.DEEP_KEY, dtype=np.float32),
                                          np.asarray(axial_region_mask == self.SUPERFICIAL_KEY, dtype=np.float32))
            else:
                axial_map = axial_region_mask == axial

            axial_map = axial_map * quant_map.volume

            for coronal in [self.MEDIAL_KEY, self.LATERAL_KEY]:
                for sagittal in [self.ANTERIOR_KEY, self.CENTRAL_KEY, self.POSTERIOR_KEY]:
                    curr_region_mask = (coronal_region_mask == coronal) * (sagittal_region_mask == sagittal) * axial_map * mask

                    # discard all values that are <= 0
                    qv_region_vals = curr_region_mask[curr_region_mask > 0]

                    c_mean = np.nanmean(qv_region_vals)
                    c_std = np.nanstd(qv_region_vals)
                    c_median = np.nanmedian(qv_region_vals)

                    row_info = [subject_pid, axial_names[axial], coronal_names[coronal], sagittal_names[sagittal],
                                c_mean, c_std, c_median]

                    pd_list.append(row_info)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [{'title': '%s deep' % qv_name, 'data': deep, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_deep.png' % qv_name, 'raw_data_filename': '%s_deep.data' % qv_name},
                {'title': '%s superficial' % qv_name, 'data': superficial, 'xlabel': 'Slice',
                 'ylabel': 'Angle (binned)', 'filename': '%s_superficial.png' % qv_name,
                 'raw_data_filename': '%s_superficial.data' % qv_name},
                {'title': '%s total' % qv_name, 'data': total, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_total.png' % qv_name,
                 'raw_data_filename': '%s_total.data' % qv_name}]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask):
        # get the largest connected component from the mask - we expect femoral cartilage to be a smooth volume
        msk = np.asarray(nlm.largest_cc(mask.volume), dtype=np.uint8)
        mask_copy = MedicalVolume(msk, mask.pixel_spacing)

        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of femoral cartilage

        Check which quantitative values (T2, T1rho, etc) are defined for femoral cartilage and analyze these
        1. Save 2D total, superficial, and deep visualization maps
        2. Save {'medial', 'lateral'}, {'anterior', 'central', 'posterior'}, {'deep', 'superficial'} data to excel
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
                ylabel = 'Angle (binned)'
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

                plt.savefig(filepath, dpi=defaults.DEFAULT_DPI)

                # Save data
                raw_data_filepath = os.path.join(q_name_dirpath, 'raw_data', q_map_data['raw_data_filename'])
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)

    def save_data(self, save_dirpath):
        super().save_data(save_dirpath)

        save_dirpath = self.__save_dirpath__(save_dirpath)

        if self.regions_mask is None:
            return

        sagital_region_mask, coronal_region_mask = self.__split_mask__()

        # Save region map - add by 1 because no key can be 0
        coronal_region_mask = (coronal_region_mask + 1) * 10
        sagital_region_mask = (sagital_region_mask + 1)
        joined_mask = coronal_region_mask + sagital_region_mask
        labels = ['medial anterior', 'medial central', 'medial posterior',
                  'lateral anterior', 'lateral central', 'lateral posterior']
        plt_dict = {'labels': labels, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)', 'title': 'Unrolled Regions'}
        img_utils.write_regions(os.path.join(save_dirpath, 'region_map.png'), joined_mask, plt_dict=plt_dict)

    def __split_mask__(self):

        assert self.ML_BOUNDARY is not None and self.ACP_BOUNDARY is not None, "medial/lateral and anterior/central/posterior boundaries should be specified"

        # split into regions
        unrolled_total, _, _ = self.unroll(np.asarray(self.__mask__.volume, dtype=np.float64))

        acp_division_unrolled = np.zeros(unrolled_total.shape)

        ac_threshold = self.ACP_BOUNDARY[0]
        cp_threshold = self.ACP_BOUNDARY[1]
        acp_division_unrolled[:ac_threshold, :] = self.ANTERIOR_KEY
        acp_division_unrolled[ac_threshold:cp_threshold, :] = self.CENTRAL_KEY
        acp_division_unrolled[cp_threshold:, :] = self.POSTERIOR_KEY

        ml_division_unrolled = np.zeros(unrolled_total.shape)
        if self.medial_to_lateral:
            ml_division_unrolled[..., :self.ML_BOUNDARY] = self.MEDIAL_KEY
            ml_division_unrolled[..., self.ML_BOUNDARY:] = self.LATERAL_KEY
        else:
            ml_division_unrolled[..., :self.ML_BOUNDARY] = self.LATERAL_KEY
            ml_division_unrolled[..., self.ML_BOUNDARY:] = self.MEDIAL_KEY

        acp_division_unrolled[np.isnan(unrolled_total)] = np.nan
        ml_division_unrolled[np.isnan(unrolled_total)] = np.nan

        return acp_division_unrolled, ml_division_unrolled
