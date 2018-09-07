import numpy as np
import math
from pydicom.tag import Tag
from scipy import optimize
from skimage.transform import resize

from utils import dicom_utils

# DESS DICOM header keys
DESS_GL_AREA_TAG = Tag(0x001910b6)
DESS_TG_TAG = Tag(0x001910b7)

# DESS constants
NUM_ECHOS = 2
T1 = 1.2
D = 1.25*1e-9


def calc_t2_map(dicom_array, ref_dicom):
    """ Calculate t2 map
    :param dicom_array: 3D numpy array in dual echo format
                        (echo 1 = dicom_array[:,:,0::2], echo 2 = dicom_array[:,:,1::2])
    :param ref_dicom: a pydicom reference/header

    :rtype: 2D numpy array with values (0, 100] and np.nans
    """
    if len(dicom_array.shape) != 3:
        raise ValueError("dicom_array must be 3D volume")

    r, c, num_slices = dicom_array.shape

    # Split echos
    subvolumes = dicom_utils.split_volume(dicom_array, 2)
    echo_1 = subvolumes[0]
    echo_2 = subvolumes[1]

    # All timing in seconds
    TR = float(ref_dicom.RepetitionTime) * 1e-3
    TE = float(ref_dicom.EchoTime) * 1e-3
    Tg = float(ref_dicom[DESS_TG_TAG].value) * 1e-6

    # Flip Angle (degree -> radians)
    alpha = math.radians(float(ref_dicom.FlipAngle))

    GlArea = float(ref_dicom[DESS_GL_AREA_TAG].value)

    Gl = GlArea / (Tg * 1e6) * 100
    gamma = 4258 * 2 * math.pi # Gamma, Rad / (G * s).
    dkL = gamma * Gl * Tg

    # Simply math
    k = math.pow((math.sin(alpha / 2)), 2) * (1 + math.exp(-TR/T1 - TR * math.pow(dkL, 2) * D)) / (1 - math.cos(alpha)* math.exp(-TR/T1 - TR * math.pow(dkL, 2) * D))

    c1 = (TR - Tg/3) * (math.pow(dkL, 2)) * D

    # T2 fit
    mask = np.ones([r, c, int(num_slices / 2)])

    ratio = mask * echo_2 / echo_1
    ratio = np.nan_to_num(ratio)

    t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

    t2map = np.nan_to_num(t2map)

    # Filter calculated T2 values that are below 0ms and over 100ms
    t2map[t2map <= 0] = np.nan
    t2map[t2map > 100] = np.nan

    return t2map


def circle_fit(x,y):
    ###
    # this function fit a circle given (x,y) scatter points in a plane.
    #
    # INPUT:
    #
    #   x................numpy array (n,) where n is the length of the array
    #   y................numpy array (n,) where n is the length of the array
    #
    # OUTPUT:
    #
    #   xc_2.............scalar, it is the coordinate x of the fitted circle
    #   yc_2.............scalar, it is the coordinate y of the fitted circle
    #   R_2..............scalar, it is the radius of the fitted circle
    ###


    # initialize the coordinate for the fitting procedure
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)
    R_2        = Ri_2.mean()
    #residu_2   = sum((Ri_2 - R_2)**2)
    #residu2_2  = sum((Ri_2**2-R_2**2)**2)

    return xc_2, yc_2, R_2


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return(rho, phi)


def get_t2_and_unroll(t2_map, mask):

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
    #   mask............................numpy array (n,n,nb_slices) which contains the segmentation mask
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

    if (t2_map.shape != mask.shape):
        raise ValueError('t2_map and mask must have same shape')

    if (len(t2_map.shape) != 3):
        raise ValueError('t2_map and mask must be 3D')

    num_slices = t2_map.shape[2]

    ## STEP 1: PROJECTING AND CYLINDRICAL FIT

    thikness_divisor = 0.5

    segmented_T2maps = np.multiply(mask, t2_map)                                                                        # apply binary mask

    #TODO: determine what clipping points should be
    segmented_T2maps = np.clip(segmented_T2maps, 0, 80)                                                                 # eliminate non physiological high T2 values

    segmented_T2maps_projected = np.max(segmented_T2maps, 2)                                                            # Project segmented T2maps on sagittal axis

    non_zero_element = np.nonzero(segmented_T2maps_projected)

    xc_fit, yc_fit, R_fit = circle_fit(non_zero_element[0], non_zero_element[1])                                        # fit a circle to projected cartilage tissue

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

