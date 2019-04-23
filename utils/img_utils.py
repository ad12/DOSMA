import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

import defaults
from data_io.fig_format import savefig


def downsample_slice(img_array, ds_factor, is_mask=False):
    '''
    
    Function Info:
        Takes in a 3D array and then downsamples in the z-direction by 
        a user-specified downsampling factor
        
    Parameters:
    -----------
    img_array : float32 array
        3D numpy array for now (xres x yres x zres)
    ds_factor : int
        Downsampling factor
    is_mask : bool
        Set to True if downsampling a mask and binarizing (unlike for image)

    Returns:
    --------
    final : float32 array
        3D numpy array for now (xres x yres x zres//ds_factor)
        
    Example:
    --------        
    input_image  = numpy.random.rand(4,4,4)
    input_mask   = (a > 0.5)*1
    output_image = downsample_slice(input_mask, ds_factor = 2, is_mask = False)
    output_mask  = downsample_slice(input_mask, ds_factor = 2, is_mask = True)

    Created By:
    -----------  
    Akshay Chaudhari, akshaysc@stanford.edu, October 16th 2018

    '''

    img_array = np.transpose(img_array, (2, 0, 1))
    L = list(img_array)

    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=0, *args)

    final = np.array([sum(x) for x in grouper(L, ds_factor)])
    final = np.transpose(final, (1, 2, 0))

    #     Binarize if it is a mask
    if is_mask is True:
        final = (final >= 1) * 1

    return final


def write_regions(filepath, arr, plt_dict=None):
    """
    Write 2D array to region image where colors correspond to the region

    All finite values should be >= 1
    nan/inf value are ignored - written as white

    :param filepath: Filepath to save image
    :param arr: The 2D numpy array to convert to region image
    :param labels: labels for unique values, default = None
    """

    if len(arr.shape) != 2:
        raise ValueError('\'arr\' must be a 2D numpy array')

    unique_vals = np.unique(arr.flatten())
    if 0 in unique_vals:
        raise ValueError('All finite values in \'arr\' must be >=1')

    unique_vals = unique_vals[np.isfinite(unique_vals)]
    num_unique_vals = len(unique_vals)

    if plt_dict is None:
        plt_dict = {'xlabel': '', 'ylabel': '', 'title': '', 'labels': None}

    labels = plt_dict['labels']
    if labels is None:
        labels = list(unique_vals)

    if len(labels) != num_unique_vals:
        raise ValueError('len(labels) != num_unique_vals - %d != %d' % (len(labels), num_unique_vals))

    cpal = sns.color_palette("pastel", num_unique_vals)

    arr_c = np.array(arr)
    arr_c = np.nan_to_num(arr_c)
    arr_c[arr_c > np.max(unique_vals)] = 0
    arr_rgb = np.ones([arr_c.shape[0], arr_c.shape[1], 3])

    plt.figure()
    plt.clf()
    custom_lines = []
    for i in range(num_unique_vals):
        unique_val = unique_vals[i]
        i0, i1 = np.where(arr_c == unique_val)
        arr_rgb[i0, i1, ...] = np.asarray(cpal[i])

        custom_lines.append(Line2D([], [], color=cpal[i], marker='o', linestyle='None',
                                   markersize=5))

    plt.xlabel(plt_dict['xlabel'])
    plt.ylabel(plt_dict['ylabel'])
    plt.title(plt_dict['title'])

    lgd = plt.legend(custom_lines, labels, loc='upper center', bbox_to_anchor=(0.5, -defaults.DEFAULT_TEXT_SPACING),
                     fancybox=True, shadow=True, ncol=3)
    plt.imshow(arr_rgb)

    savefig(filepath, dpi=defaults.DEFAULT_DPI, bbox_extra_artists=(lgd,), bbox_inches='tight')
