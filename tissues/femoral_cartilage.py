import numpy as np
from tissues.tissue import Tissue

from utils import io_utils
from utils.geometry_utils import circle_fit, cart2pol
from skimage.transform import resize
import scipy.io as sio
import pandas as pd
import os

__MASK_KEY__ = 'mask'
__T2_DATA_KEY__ = 't2_data'
__T1_RHO_DATA_KEY__ = 't1_rho_data'
__T2_STAR_DATA_KEY__ = 't2_star_data'
QUANTITATIVE_VALUES_KEYS = [__T1_RHO_DATA_KEY__, __T2_DATA_KEY__, __T2_STAR_DATA_KEY__]


class FemoralCartilage(Tissue):
    ID = 1
    NAME = 'fc'

    # Coronal Keys
    ANTERIOR_KEY = 0
    CENTRAL_KEY = 1
    POSTERIOR_KEY = 2
    CORONAL_KEYS = [ANTERIOR_KEY, CENTRAL_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 0
    LATERAL_KEY = 1
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    def __init__(self, weights_dir = None):
        super().__init__(weights_dir=weights_dir)
        self.regions_mask = None

    def unroll(self, map):

        ## UNROLLING CARTILAGE T2 MAPS
        #
        # the function applies a segmentation mask to the T2 maps. It then fits a circle to the sagittal projection of the
        # 3D segmentation mask.
        # Slice per slice the fitted circle is used to put T2 values of each slice into degree bins. In this way the tissue
        # is unrolled. The cartilage is then divided into deep and superficial cartilage.
        # As final step, the arrays are resized to fit [512,512] resolution.
        #
        # INPUT:
        #   TODO: by default t2 maps have nan values - we should handle these by clipping possibly?
        #   t2_map..........................numpy array (n,n,nb_slices) which contains the T2 map
        #
        # OUTPUT:
        #
        #   Unrolled_Cartilage_res..........numpy array (n,nb_bins) which contains the unrolled cartilage T2 maps...
        #                                   ...considering ALL the layers
        #   Sup_layer_res...................numpy array (n,nb_bins) which contains the unrolled cartilage T2 maps...
        #                                   ...considering the SUPERFICIAL layers
        #   Deep_layer_res..................numpy array (n,nb_bins) which contains the unrolled cartilage T2 maps...
        #                                   ...considering the DEEP layers
        ###

        mask = self.mask

        if (map.shape != mask.shape):
            raise ValueError('t2_map and mask must have same shape')

        if (len(map.shape) != 3):
            raise ValueError('t2_map and mask must be 3D')

        num_slices = map.shape[2]

        ## STEP 1: PROJECTING AND CYLINDRICAL FIT

        thikness_divisor = 0.5

        segmented_T2maps = np.multiply(mask, map)  # apply binary mask

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

                Unrolled_Cartilage[i, np.int((angle + 180) / 5 + 1)] = np.mean(binned_result[:, 2], axis=0)
                Sup_layer[i, np.int((angle + 180) / 5 + 1)] = np.mean(binned_super[:, 2], axis=0)
                Deep_layer[i, np.int((angle + 180) / 5 + 1)] = np.mean(binned_deep[:, 2], axis=0)

        ## STEP 3: RESIZE DATA TO [512,512] DIMENSION
        # TODO: is resizing required? can we keep them in the same dimensions as the input: yes we can :)
        # Unrolled_Cartilage_res = resize(Unrolled_Cartilage, (512, 512), order=1, preserve_range=True)
        # Sup_layer_res = resize(Sup_layer, (512, 512), order=1, preserve_range=True)
        # Deep_layer_res = resize(Deep_layer, (512, 512), order=1, preserve_range=True)

        total_cartilage_unrolled = np.transpose(Unrolled_Cartilage)
        sup_layer_unrolled = np.transpose(Sup_layer)
        deep_layer_unrolled = np.transpose(Deep_layer)

        return total_cartilage_unrolled, sup_layer_unrolled, deep_layer_unrolled

        def split_regions(self, unrolled_quantitative_map):
            # WARNING: this method has to be called after the unroll method.

            # create unrolled mask from unrolled map
            unrolled_mask_indexes = np.nonzero(unrolled_quantitative_map)
            unrolled_mask = np.zeros((unrolled_quantitative_map.shape[0], unrolled_quantitative_map.shape[1]))
            unrolled_mask[unrolled_mask_indexes] = 1
            unrolled_mask[np.where(unrolled_mask < 1)] = 3

            # find the center of mass of the unrolled mask
            center_of_mass = ndimage.measurements.center_of_mass(unrolled_mask)

            lateral_mask = np.transpose(np.copy(unrolled_mask))[:, 0:np.int(center_of_mass[1])]
            medial_mask = np.transpose(np.copy(unrolled_mask))[:, np.int(center_of_mass[1]):]

            lateral_mask[np.where(lateral_mask < 3)] = self.LATERAL_KEY
            medial_mask[np.where(medial_mask < 3)] = self.MEDIAL_KEY

            ml_mask = np.concatenate((lateral_mask, medial_mask), axis=1)

            # Split map in anterior, central and posterior regions
            anterior_mask = np.transpose(np.copy(unrolled_mask))[0:np.int(center_of_mass[0]), :]
            central_mask = np.transpose(np.copy(unrolled_mask))[
                           np.int(center_of_mass[0]):np.int(center_of_mass[0]) + 10, :]
            posterior_mask = np.transpose(np.copy(unrolled_mask))[np.int(center_of_mass[0]) + 10:, :]

            anterior_mask[np.where(anterior_mask < 3)] = self.ANTERIOR_KEY
            posterior_mask[np.where(posterior_mask < 3)] = self.POSTERIOR_KEY
            central_mask[np.where(central_mask < 3)] = self.CENTRAL_KEY

            acp_mask = np.concatenate((anterior_mask, central_mask, posterior_mask), axis=0)

            self.regions_mask = np.concatenate((ml_mask, acp_mask), axis=2)

    def calc_quant_vals(self, quant_map, map_type):
        """
        Calculate quantitative values and store in excel file
        :param quant_map: The 3D volume of quantitative values (np.nan for all pixels that are not accurate)
        :param map_type: A QuantitativeValue instance
        :return:
        """
        if self.mask is None or self.regions_mask is None:
            raise ValueError('Please call split_regions first')

        mask = self.mask
        coronal_region_mask = self.regions_mask(..., 0)
        sagital_region_mask = self.regions_mask(..., 1)

        total, deep, superficial = self.unroll(self.mask * quant_map)
        # TODO: identify pixels in deep and superficial that are anterior/central/posterior and medial/lateral
        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
                         ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
                         ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]
        tissue_values = []
        for axial_map in [total, deep, superficial]:
            for coronal in [self.MEDIAL_KEY, self.LATERAL_KEY]:
                coronal_list = []
                for sagital in [self.ANTERIOR_KEY, self.CENTRAL_KEY, self.POSTERIOR_KEY]:
                    curr_region_mask = (coronal_region_mask == coronal) * (sagital_region_mask == sagital) * axial_map

                    # discard all values that are 0
                    c_mean = np.nanmean(curr_region_mask)
                    c_std = np.nanstd(curr_region_mask)
                    coronal_list.append('%0.2f +/- %0.2f' % (c_mean, c_std))

                tissue_values.append(coronal_list)

        depth_keys = np.array(['total', 'total', 'deep', 'deep', 'superficial', 'superficial'])
        coronal_keys = np.array(['medial', 'lateral'] * 3)
        sagital_keys = ['anterior', 'central', 'posterior']
        df = pd.DataFrame(data=np.transpose(tissue_values), index=sagital_keys, columns=pd.MultiIndex.from_tuples(zip(depth_keys, coronal_keys)))

        self.__store_quant_vals__(quant_map, df, map_type)


