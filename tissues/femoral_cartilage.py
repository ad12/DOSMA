import numpy as np
import os
from tissues.tissue import Tissue

from utils import io_utils
from utils.geometry_utils import circle_fit, cart2pol
import scipy.ndimage as sni
import pandas as pd

import nipy.labs.mask as nlm

from utils.quant_vals import QuantitativeValues
import matplotlib.pyplot as plt

import defaults
import warnings

BOUNDS = {QuantitativeValues.T2: 100.0,
          QuantitativeValues.T1_RHO: 150.0,
          QuantitativeValues.T2_STAR: 100.0}

class FemoralCartilage(Tissue):
    ID = 1
    STR_ID = 'fc'
    FULL_NAME = 'femoral cartilage'

    ORIENTATION = 'RIGHT'

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

    def unroll(self, qv_map):

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

        if (qv_map.shape != mask.shape):
            raise ValueError('t2_map and mask must have same shape')

        if (len(qv_map.shape) != 3):
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
        # WARNING: this method has to be called after the unroll method.
        import matplotlib.pyplot as plt

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

        if self.ORIENTATION == 'RIGHT':
            ml_mask = np.concatenate((lateral_mask, medial_mask), axis=1)
        else:
            ml_mask = np.concatenate((medial_mask, lateral_mask), axis=1)

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
        Calculate quantitative values and store in excel file
        :param quant_map: The 3D volume of quantitative values (np.nan for all pixels that are not accurate)
        :param map_type: A QuantitativeValue instance
        :return:
        """
        if self.mask is None:
            raise ValueError('Please initialize mask')

        total, superficial, deep = self.unroll(quant_map)

        assert total.shape == deep.shape
        assert deep.shape == superficial.shape

        if self.regions_mask is None:
            self.split_regions(total)

        coronal_region_mask = self.regions_mask[..., 0]
        sagital_region_mask = self.regions_mask[..., 1]

        # TODO: identify pixels in deep and superficial that are anterior/central/posterior and medial/lateral
        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'],
                         ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP'],
                         ['TMA', 'TMC', 'TMP'], ['TLA', 'TLC', 'TLP']]
        tissue_values = []
        for axial_map in [deep, superficial, total]:
            for coronal in [self.MEDIAL_KEY, self.LATERAL_KEY]:
                coronal_list = []
                for sagital in [self.ANTERIOR_KEY, self.CENTRAL_KEY, self.POSTERIOR_KEY]:
                    curr_region_mask = (coronal_region_mask == coronal) * (sagital_region_mask == sagital) * axial_map

                    curr_region_mask[curr_region_mask==0] = np.nan
                    # discard all values that are 0
                    c_mean = np.nanmean(curr_region_mask)
                    c_std = np.nanstd(curr_region_mask)
                    c_median = np.nanmedian(curr_region_mask)
                    coronal_list.append('%0.5f +/- %0.5f, %0.5f' % (c_mean, c_std, c_median))

                tissue_values.append(coronal_list)

        depth_keys = np.array(['deep', 'deep', 'superficial', 'superficial', 'total', 'total'])
        coronal_keys = np.array(['medial', 'lateral'] * 3)
        sagital_keys = ['anterior', 'central', 'posterior']
        df = pd.DataFrame(data=np.transpose(tissue_values), index=sagital_keys, columns=pd.MultiIndex.from_tuples(zip(depth_keys, coronal_keys)))

        qv_name = map_type.name
        maps = [{'title': '%s deep' % qv_name, 'data': deep, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)', 'filename': '%s_deep.png' % qv_name},
                {'title': '%s superficial' % qv_name, 'data': superficial, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)', 'filename': '%s_superficial.png' % qv_name},
                {'title': '%s total' % qv_name, 'data': total, 'xlabel': 'Slice', 'ylabel': 'Angle (binned)', 'filename': '%s_total.png' % qv_name}]

        self.__store_quant_vals__(maps, df, map_type)

    def set_mask(self, mask, pixel_spacing):
        mask = np.asarray(nlm.largest_cc(mask), dtype=np.uint8)
        self.regions_mask = None
        super().set_mask(mask, pixel_spacing)

    def __save_quant_data__(self, dirpath):
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
                if defaults.FIX_VISUALIZATION_BOUNDS:
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
