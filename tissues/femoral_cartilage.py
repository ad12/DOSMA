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

    # Axial keys
    DEEP_KEY = 'deep'
    SUPERFICIAL_KEY = 'superficial'
    AXIAL_KEYS = [DEEP_KEY, SUPERFICIAL_KEY]

    # Coronal Keys
    ANTERIOR_KEY = 'anterior'
    CENTRAL_KEY = 'central'
    POSTERIOR_KEY = 'posterior'
    CORONAL_KEYS = [ANTERIOR_KEY, CENTRAL_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 'medial'
    LATERAL_KEY = 'lateral'
    SAGGITAL_KEYS = [ANTERIOR_KEY, CENTRAL_KEY, POSTERIOR_KEY]


    def __init__(self, weights_dir=None):
        super().__init__(weights_dir=weights_dir)
        self.regions = {}


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
        # TODO: is resizing required? can we keep them in the same dimensions as the input
        Unrolled_Cartilage_res = resize(Unrolled_Cartilage, (512, 512), order=1, preserve_range=True)
        Sup_layer_res = resize(Sup_layer, (512, 512), order=1, preserve_range=True)
        Deep_layer_res = resize(Deep_layer, (512, 512), order=1, preserve_range=True)

        return Unrolled_Cartilage_res, Sup_layer_res, Deep_layer_res

    def split_regions(self, mask):
        #TODO: implement spliting region
        pass

    def calc_quant_vals(self, quant_map, map_type):
        mask = self.mask
        unrolled, deep, superficial = self.unroll(quant_map)
        # TODO: identify pixels in deep and superficial that are anterior/central/posterior and medial/lateral
        # Replace strings with values - eg. DMA = 'deep, medial, anterior'
        tissue_values = [['DMA', 'DMC', 'DMP'], ['DLA', 'DLC', 'DLP'], ['SMA', 'SMC', 'SMP'], ['SLA', 'SLC', 'SLP']]
        depth_keys = np.array(['deep', 'deep', 'superficial', 'superficial'])
        coronal_keys = np.array(['medial', 'lateral'] * 2)
        sagital_keys = ['anterior', 'central', 'posterior']
        df = pd.DataFrame(data=np.transpose(tissue_values), index=sagital_keys, columns=pd.MultiIndex.from_tuples(zip(depth_keys, coronal_keys)))

        self.__store_quant_vals__(quant_map, df, map_type)


    def save_data(self, dirpath):
        data = dict()
        data.update(self.quant_vals)
        data.update({__MASK_KEY__: self.mask})

        # Save to h5 file
        io_utils.save_h5(os.path.join(dirpath, self.__data_filename__()), data)


