import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import nipy.labs.mask as nlm
import numpy as np
import pandas as pd
import scipy.ndimage as sni

import defaults
from data_io.format_io import ImageDataFormat
from data_io.med_volume import MedicalVolume
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from tissues.tissue import Tissue
from utils import io_utils, img_utils
from utils.geometry_utils import circle_fit, cart2pol
from utils.quant_vals import QuantitativeValues

from data_io.fig_format import savefig

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

    # Keys correspond to integer representing bit location for each region
    # bit string: 'T D S M L A C P' (stored as integer)
    # Coronal Keys
    POSTERIOR_KEY = 2 ** 0
    CENTRAL_KEY = 2 ** 1
    ANTERIOR_KEY = 2 ** 2
    CORONAL_KEYS = [POSTERIOR_KEY, CENTRAL_KEY, ANTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 2 ** 3
    LATERAL_KEY = 2 ** 4
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    # Axial Keys
    DEEP_KEY = 2 ** 5
    SUPERFICIAL_KEY = 2 ** 6
    TOTAL_AXIAL_KEY = 2 ** 7
    AXIAL_KEYS = [DEEP_KEY, SUPERFICIAL_KEY, TOTAL_AXIAL_KEY]

    # Do not change order of below. Order reflects order of CORONAL_KEYS, SAGGITAL_KEYS, AXIAL_KEYS
    AXIAL_NAMES = ['deep', 'superficial', 'total']
    SAGITTAL_NAMES = ['medial', 'lateral']
    CORONAL_NAMES = ['posterior', 'central', 'anterior']

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

        mask = mask * np.nan_to_num(base_map)

        height, width, num_slices = mask.shape

        # STEP 1: PROJECTING AND CYLINDRICAL FIT
        segmented_t2maps_projected = np.max(mask, 2)  # Project segmented T2maps on sagittal axis
        non_zero_element = np.nonzero(segmented_t2maps_projected)

        xc_fit, yc_fit, R_fit = circle_fit(non_zero_element[1],
                                           non_zero_element[0])  # fit a circle to projected cartilage tissue

        # STEP 2: SLICE BY SLICE BINNING
        yv, xv = np.meshgrid(range(height), range(width), indexing='ij')

        rho, theta = cart2pol(xv - xc_fit, yc_fit - yv)
        theta = (theta >= 90) * (theta - 360) + (theta < 90) * theta  # range: [-270, 90)

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

        regions_volume = np.asarray(np.zeros(mask.shape), dtype=np.uint16)

        # anterior/central/posterior division
        # Central region occupies middle 30 degrees, anterior on left, posterior on right
        anterior_region = self.ANTERIOR_KEY * (theta < -105)
        central_region = self.CENTRAL_KEY * np.logical_and((theta >= -105), (theta < -75))
        posterior_region = self.POSTERIOR_KEY * (theta >= -75)
        acp_map = anterior_region + central_region + posterior_region
        acp_volume = np.asarray(np.stack([acp_map] * num_slices, axis=-1), dtype=np.uint16)
        regions_volume += acp_volume

        # medial/lateral division
        # take into account scanning direction
        center_of_mass = sni.measurements.center_of_mass(mask)
        com_slicewise = center_of_mass[-1]
        ml_volume = np.asarray(np.zeros(mask.shape), dtype=np.uint16)

        if self.medial_to_lateral:
            ml_volume[..., :int(np.ceil(com_slicewise))] = self.MEDIAL_KEY
            ml_volume[..., int(np.ceil(com_slicewise)):] = self.LATERAL_KEY
        else:
            ml_volume[..., :int(np.ceil(com_slicewise))] = self.LATERAL_KEY
            ml_volume[..., int(np.ceil(com_slicewise)):] = self.MEDIAL_KEY
        regions_volume += ml_volume

        # deep/superficial division
        rho_volume = np.stack([rho] * num_slices, axis=-1)
        deep_volume = (rho_volume <= rhos_threshold_volume) * self.DEEP_KEY
        superficial_volume = (rho_volume >= rhos_threshold_volume) * self.SUPERFICIAL_KEY
        ds_volume = np.asarray(deep_volume + superficial_volume + self.TOTAL_AXIAL_KEY, dtype=np.uint16)

        regions_volume += ds_volume
        ml_boundary = int(np.ceil(com_slicewise))
        acp_boundary = [int(np.floor((-105 - THETA_MIN) / DTHETA)), int(np.floor((-75 - THETA_MIN) / DTHETA))]

        return regions_volume, theta_bins, ml_boundary, acp_boundary

    def unroll(self, qv_map, regions_mask, theta_bins):
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

        # assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        num_slices = qv_map.shape[-1]

        qv_map = np.nan_to_num(qv_map)
        qv_map = np.multiply(mask, qv_map)  # apply binary mask
        qv_map[qv_map <= 0] = np.nan  # wherever qv_map is 0, either no cartilage or qv=0 ms, which is impractical

        # theta_bins = self.theta_bins  # binning with theta

        # regions_mask = self.regions_mask

        Unrolled_Cartilage = np.zeros([NB_BINS, num_slices])
        Sup_layer = np.zeros([NB_BINS, num_slices])
        Deep_layer = np.zeros([NB_BINS, num_slices])

        for slice_ind in range(num_slices):
            qv_slice = qv_map[..., slice_ind]
            curr_slice = regions_mask[..., slice_ind]

            # if slice is all NaNs, then don't analyze
            if np.sum(np.isnan(qv_slice)) == qv_slice.shape[0] * qv_slice.shape[1]:
                continue

            for curr_bin in range(NB_BINS):
                qv_bin = qv_slice[theta_bins == curr_bin]
                if np.sum(np.isnan(qv_bin)) == len(qv_bin):
                    continue

                Unrolled_Cartilage[curr_bin, slice_ind] = np.nanmean(qv_bin)

                qv_superficial = qv_slice[np.logical_and(theta_bins == curr_bin,
                                                         self.__binarize_region_mask__(curr_slice,
                                                                                       self.SUPERFICIAL_KEY))]
                qv_deep = qv_slice[
                    np.logical_and(theta_bins == curr_bin, self.__binarize_region_mask__(curr_slice, self.DEEP_KEY))]

                qv_superficial = np.nan_to_num(qv_superficial)
                qv_deep = np.nan_to_num(qv_deep)

                qv_sup_mean = np.mean(qv_superficial[qv_superficial > 0])
                qv_deep_mean = np.mean(qv_deep[qv_deep > 0])
                Sup_layer[curr_bin, slice_ind] = qv_sup_mean
                Deep_layer[curr_bin, slice_ind] = qv_deep_mean

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

        :param quant_map: The 3D volumes of quantitative values (np.nan for all pixels that are not accurate)
        :param map_type: A QuantitativeValue instance
        """

        super().__calc_quant_vals__(quant_map, map_type)

        # assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"

        # We have to call this every time we load a new quantitative map
        # mask = segmentation_mask * clipped_quant_map
        regions_mask, theta_bins, _, _ = self.split_regions(quant_map.volume)

        total, superficial, deep = self.unroll(quant_map.volume, regions_mask, theta_bins)

        assert total.shape == deep.shape
        assert deep.shape == superficial.shape

        # regions_mask = self.regions_mask
        mask = self.__mask__.volume

        subject_pid = self.pid
        pd_header = ['Subject', 'Location', 'Side', 'Region', 'Mean', 'Std', 'Median', '# Voxels']
        pd_list = []

        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        # tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
        #                  ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
        #                  ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]
        # tissue_values = []

        for axial_ind in range(len(self.AXIAL_KEYS)):
            axial = self.AXIAL_KEYS[axial_ind]

            for sagittal_ind in range(len(self.SAGGITAL_KEYS)):
                sagittal = self.SAGGITAL_KEYS[sagittal_ind]
                for coronal_ind in range(len(self.CORONAL_KEYS)):
                    coronal = self.CORONAL_KEYS[coronal_ind]

                    curr_region_mask = self.__binarize_region_mask__(regions_mask, (axial | coronal | sagittal))
                    curr_region_mask = curr_region_mask * mask * quant_map.volume

                    # discard all values that are <= 0
                    qv_region_vals = curr_region_mask[curr_region_mask > 0]

                    num_voxels = len(qv_region_vals)

                    c_mean = np.nanmean(qv_region_vals)
                    c_std = np.nanstd(qv_region_vals)
                    c_median = np.nanmedian(qv_region_vals)

                    row_info = [subject_pid,
                                self.AXIAL_NAMES[axial_ind], self.SAGITTAL_NAMES[sagittal_ind],
                                self.CORONAL_NAMES[coronal_ind],
                                c_mean, c_std, c_median, num_voxels]

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

    def set_mask(self, mask: MedicalVolume):
        # get the largest connected component from the mask - we expect femoral cartilage to be a smooth volumes
        msk = np.asarray(nlm.largest_cc(mask.volume), dtype=np.uint8)
        mask_copy = deepcopy(mask)
        mask_copy.volume = msk

        super().set_mask(mask_copy)

        self.regions_mask, self.theta_bins, self.ML_BOUNDARY, self.ACP_BOUNDARY = self.split_regions(
            self.__mask__.volume)

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

                savefig(filepath, dpi=defaults.DEFAULT_DPI)

                # Save data
                raw_data_filepath = os.path.join(q_name_dirpath, 'raw_data', q_map_data['raw_data_filename'])
                io_utils.save_pik(raw_data_filepath, data_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)

    def save_data(self, save_dirpath, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        super().save_data(save_dirpath, data_format=data_format)

        save_dirpath = self.__save_dirpath__(save_dirpath)

        if self.regions_mask is None:
            return

        sagital_region_mask, coronal_region_mask = self.__split_mask__()

        # Save region map - add by 1 because no key can be 0
        coronal_region_mask = (coronal_region_mask + 1) * 10
        sagital_region_mask = (sagital_region_mask + 1)
        joined_mask = coronal_region_mask + sagital_region_mask
        labels = ['medial posterior', 'medial central', 'medial anterior',
                  'lateral posterior', 'lateral central', 'lateral anterior']
        plt_dict = {'labels': labels, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)', 'title': 'Unrolled Regions'}
        img_utils.write_regions(os.path.join(save_dirpath, 'region_map.png'), joined_mask, plt_dict=plt_dict)

    def __binarize_region_mask__(self, region_mask, roi):
        return np.asarray(np.bitwise_and(region_mask, roi) == roi, dtype=np.bool)

    def __split_mask__(self):

        assert self.ML_BOUNDARY is not None and self.ACP_BOUNDARY is not None, "medial/lateral and anterior/central/posterior boundaries should be specified"

        # split into regions
        unrolled_total, _, _ = self.unroll(np.asarray(self.__mask__.volume, dtype=np.float32), self.regions_mask,
                                           self.theta_bins)

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
