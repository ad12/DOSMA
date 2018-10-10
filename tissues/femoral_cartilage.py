import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nipy.labs.mask as nlm
import numpy as np
import pandas as pd
import scipy.ndimage as sni

import defaults
from med_objects.med_volume import MedicalVolume
from tissues.tissue import Tissue
from utils import io_utils
from utils.geometry_utils import circle_fit, cart2pol
from utils.quant_vals import QuantitativeValues

BOUNDS = {QuantitativeValues.T2: 60.0,
          QuantitativeValues.T1_RHO: 100.0,
          QuantitativeValues.T2_STAR: 50.0}


class FemoralCartilage(Tissue):
    """Handles analysis for femoral cartilage"""
    ID = 1
    STR_ID = 'fc'
    FULL_NAME = 'femoral cartilage'

    # Coronal Keys
    ANTERIOR_KEY = 0
    CENTRAL_KEY = 1
    POSTERIOR_KEY = 2
    CORONAL_KEYS = [ANTERIOR_KEY, CENTRAL_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 0
    LATERAL_KEY = 1
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        """
        :param weights_dir: Directory to weights files
        :param medial_to_lateral: True or False, if false, then lateral to medial
        """
        super().__init__(weights_dir=weights_dir)

        self.regions_mask = None
        self.medial_to_lateral = medial_to_lateral

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

        num_slices = qv_map.shape[-1]

        ## STEP 1: PROJECTING AND CYLINDRICAL FIT

        thikness_divisor = 0.5

        qv_map = np.nan_to_num(qv_map)

        segmented_T2maps = np.multiply(mask, qv_map)  # apply binary mask

        segmented_T2maps_projected = np.max(segmented_T2maps, 2)  # Project segmented T2maps on sagittal axis

        non_zero_element = np.nonzero(segmented_T2maps_projected)

        xc_fit, yc_fit, R_fit = circle_fit(non_zero_element[0],
                                           non_zero_element[1])  # fit a circle to projected cartilage tissue

        ## STEP 2: SLICE BY SLI2E BINNING

        nb_bins = 72

        Unrolled_Cartilage = np.float32(np.zeros((num_slices, nb_bins)))

        Sup_layer = np.float32(np.zeros((num_slices, nb_bins)))
        Deep_layer = np.float32(np.zeros((num_slices, nb_bins)))

        for i in np.array(range(num_slices)):

            segmented_T2maps_slice = segmented_T2maps[:, :, i]

            if np.max(np.max(segmented_T2maps_slice)) == 0:
                continue

            non_zero_slice_element = np.nonzero(segmented_T2maps_slice)
            non_zero_T2_slice_values = segmented_T2maps_slice[segmented_T2maps_slice > 0]
            dim = non_zero_T2_slice_values.shape[0]

            x_index_c = non_zero_slice_element[0] - xc_fit
            y_index_c = non_zero_slice_element[1] - yc_fit

            rho, theta_rad = cart2pol(x_index_c, y_index_c)

            theta = theta_rad * (180 / np.pi)

            polar_coords = np.concatenate(
                (theta.reshape(dim, 1), rho.reshape(dim, 1), non_zero_T2_slice_values.reshape(dim, 1)), axis=1)

            angles = np.linspace(-180, 175, num=72)

            for angle in angles:
                bottom_bin = angle
                top_bin = angle + 5

                splice_matrix = np.where((polar_coords[:, 0] > bottom_bin) & (polar_coords[:, 0] <= top_bin))

                binned_result = polar_coords[splice_matrix[0], :]

                if binned_result.size == 0:
                    continue

                max_radius = np.max(binned_result[:, 1])
                min_radius = np.min(binned_result[:, 1])

                cart_thickness = max_radius - min_radius

                rad_division = min_radius + cart_thickness * thikness_divisor

                splice_deep = np.where(binned_result[:, 1] <= rad_division)
                binned_deep = binned_result[splice_deep]

                splice_super = np.where(binned_result[:, 1] >= rad_division)
                binned_super = binned_result[splice_super]

                Unrolled_Cartilage[i, np.int((angle + 180) / 5)] = np.mean(binned_result[:, 2], axis=0)
                Sup_layer[i, np.int((angle + 180) / 5)] = np.mean(binned_super[:, 2], axis=0)
                Deep_layer[i, np.int((angle + 180) / 5)] = np.mean(binned_deep[:, 2], axis=0)

        Unrolled_Cartilage[Unrolled_Cartilage == 0] = np.nan
        Sup_layer[Sup_layer == 0] = np.nan
        Deep_layer[Deep_layer == 0] = np.nan

        total_cartilage_unrolled = np.transpose(Unrolled_Cartilage)
        sup_layer_unrolled = np.transpose(Sup_layer)
        deep_layer_unrolled = np.transpose(Deep_layer)

        return total_cartilage_unrolled, sup_layer_unrolled, deep_layer_unrolled

    def split_regions(self, unrolled_quantitative_map):
        """Split 2D MxN unrolled quantitative map into regions

        Generates a MxNx2 mask of medial/lateral pixels and anterior/central/posterior pixels
        This method must be called after the unrolled method

        :param unrolled_quantitative_map:
        :return: a MxNx2 numpy array with values corresponding to regions
        """

        # create unrolled mask from unrolled map
        unrolled_mask_indexes = np.nonzero(unrolled_quantitative_map)
        unrolled_mask = np.zeros((unrolled_quantitative_map.shape[0], unrolled_quantitative_map.shape[1]))
        unrolled_mask[unrolled_mask_indexes] = 1
        unrolled_mask[np.where(unrolled_mask < 1)] = 3

        # find the center of mass of the unrolled mask
        center_of_mass = sni.measurements.center_of_mass(unrolled_mask)

        lateral_mask = np.copy(unrolled_mask)[:, 0:np.int(np.around(center_of_mass[1]))]
        medial_mask = np.copy(unrolled_mask)[:, np.int(np.around(center_of_mass[1])):]

        lateral_mask[np.where(lateral_mask < 3)] = self.LATERAL_KEY
        medial_mask[np.where(medial_mask < 3)] = self.MEDIAL_KEY

        if self.medial_to_lateral:
            ml_mask = np.concatenate((medial_mask, lateral_mask), axis=1)
        else:
            ml_mask = np.concatenate((lateral_mask, medial_mask), axis=1)

        # Split map in anterior, central and posterior regions
        anterior_mask = np.copy(unrolled_mask)[0:np.int(center_of_mass[0]), :]
        central_mask = np.copy(unrolled_mask)[np.int(center_of_mass[0]):np.int(center_of_mass[0]) + 10, :]
        posterior_mask = np.copy(unrolled_mask)[np.int(center_of_mass[0]) + 10:, :]

        anterior_mask[np.where(anterior_mask < 3)] = self.ANTERIOR_KEY
        posterior_mask[np.where(posterior_mask < 3)] = self.POSTERIOR_KEY
        central_mask[np.where(central_mask < 3)] = self.CENTRAL_KEY

        acp_mask = np.concatenate((anterior_mask, central_mask, posterior_mask), axis=0)

        assert ml_mask.shape == acp_mask.shape

        ml_mask = ml_mask[..., np.newaxis]
        acp_mask = acp_mask[..., np.newaxis]

        self.regions_mask = np.concatenate((ml_mask, acp_mask), axis=2)

        assert (self.regions_mask[..., 0] == ml_mask[..., 0]).all()
        assert (self.regions_mask[..., 1] == acp_mask[..., 0]).all()

    def calc_quant_vals(self, quant_map, map_type):
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

        super().calc_quant_vals(quant_map, map_type)

        if self.__mask__ is None:
            raise ValueError('Please initialize mask')

        total, superficial, deep = self.unroll(quant_map.volume)

        assert total.shape == deep.shape
        assert deep.shape == superficial.shape

        if self.regions_mask is None:
            self.split_regions(total)

        coronal_region_mask = self.regions_mask[..., 0]
        sagital_region_mask = self.regions_mask[..., 1]

        subject_pid = self.pid
        pd_header = ['Subject', 'Location', 'Side', 'Region', 'Mean', 'Std', 'Median']
        pd_list = []

        # TODO: identify pixels in deep and superficial that are anterior/central/posterior and medial/lateral
        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        # tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
        #                  ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
        #                  ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]
        # tissue_values = []
        axial_data = [deep, superficial, total]

        axial_names = ['deep', 'superficial', 'total']
        coronal_names = ['medial', 'lateral']
        sagittal_names = ['anterior', 'central', 'posterior']

        if self.medial_to_lateral:
            coronal_direction = [self.MEDIAL_KEY, self.LATERAL_KEY]
        else:
            coronal_direction = [self.MEDIAL_KEY, self.LATERAL_KEY]

        for axial in range(3):
            axial_map = axial_data[axial]
            for coronal in coronal_direction:
                for sagittal in [self.ANTERIOR_KEY, self.CENTRAL_KEY, self.POSTERIOR_KEY]:
                    curr_region_mask = (coronal_region_mask == coronal) * (sagital_region_mask == sagittal) * axial_map

                    curr_region_mask[curr_region_mask == 0] = np.nan
                    # discard all values that are 0
                    c_mean = np.nanmean(curr_region_mask)
                    c_std = np.nanstd(curr_region_mask)
                    c_median = np.nanmedian(curr_region_mask)

                    row_info = [subject_pid, axial_names[axial], coronal_names[coronal], sagittal_names[sagittal],
                                c_mean, c_std, c_median]

                    pd_list.append(row_info)

        df = pd.DataFrame(pd_list, columns=pd_header)
        qv_name = map_type.name
        maps = [{'title': '%s deep' % qv_name, 'data': deep, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_deep.png' % qv_name},
                {'title': '%s superficial' % qv_name, 'data': superficial, 'xlabel': 'Slice',
                 'ylabel': 'Angle (binned)', 'filename': '%s_superficial.png' % qv_name},
                {'title': '%s total' % qv_name, 'data': total, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)',
                 'filename': '%s_total.png' % qv_name}]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask):
        assert type(mask) is MedicalVolume, "mask for femoral cartilage must be of type MedicalVolume"
        msk = np.asarray(nlm.largest_cc(mask.volume), dtype=np.uint8)
        mask_copy = MedicalVolume(msk, mask.pixel_spacing)

        self.regions_mask = None

        super().set_mask(mask_copy)

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
                plt.colorbar()

                plt.savefig(filepath)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)
