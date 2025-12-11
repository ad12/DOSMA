import itertools

import numpy as np
import seaborn as sns

from dosma import defaults

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ["downsample_slice", "write_regions"]


def downsample_slice(img_array, ds_factor, is_mask=False):
    """
    Takes in a 3D array and then downsamples in the z-direction by a
    user-specified downsampling factor.

    Args:
        img_array (np.ndarray): 3D numpy array for now (xres x yres x zres)
        ds_factor (int): Downsampling factor
        is_mask (:obj:`bool`, optional): If ``True``, ``img_array`` is a mask and will be binarized
            after downsampling. Defaults to `False`.

    Returns:
        np.ndarray: 3D numpy array of dimensions (xres x yres x zres//ds_factor)

    Examples:
        >>> input_image  = numpy.random.rand(4,4,4)
        >>> input_mask   = (a > 0.5) * 1.0
        >>> output_image = downsample_slice(input_mask, ds_factor = 2, is_mask = False)
        >>> output_mask  = downsample_slice(input_mask, ds_factor = 2, is_mask = True)
    """

    img_array = np.transpose(img_array, (2, 0, 1))
    L = list(img_array)

    def grouper(iterable, n):
        args = [iter(iterable)] * n  # noqa B026
        return itertools.zip_longest(fillvalue=0, *args)  # noqa B026

    final = np.array([sum(x) for x in grouper(L, ds_factor)])
    final = np.transpose(final, (1, 2, 0))

    # Binarize if it is a mask.
    if is_mask is True:
        final = (final >= 1) * 1

    return final


def write_regions(file_path, arr, plt_dict=None):
    """Write 2D array to region image where colors correspond to the region.

    All finite values should be >= 1.
    nan/inf value are ignored - written as white.

    Args:
        file_path (str): File path to save image.
        arr (np.ndarray): The 2D numpy array to convert to region image.
            Unique non-zero values correspond to different regions.
            Values that are `0` or `np.nan` will be written as white pixels.
        plt_dict (:obj:`dict`, optional): Dictionary of values to use when plotting with
            ``matplotlib.pyplot``. Keys are strings like `xlabel`, `ylabel`, etc.
            Use Key `labels` to specify a mapping from unique non-zero values in the array
            to names for the legend.
    """

    if len(arr.shape) != 2:
        raise ValueError("`arr` must be a 2D numpy array")

    unique_vals = np.unique(arr.flatten())
    if 0 in unique_vals:
        raise ValueError("All finite values in `arr` must be >=1")

    unique_vals = unique_vals[np.isfinite(unique_vals)]
    num_unique_vals = len(unique_vals)

    plt_dict_int = {"xlabel": "", "ylabel": "", "title": "", "labels": None}
    if plt_dict:
        plt_dict_int.update(plt_dict)
    plt_dict = plt_dict_int

    labels = plt_dict["labels"]
    if labels is None:
        labels = list(unique_vals)

    if len(labels) != num_unique_vals:
        raise ValueError(
            "len(labels) != num_unique_vals - %d != %d" % (len(labels), num_unique_vals)
        )

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

        custom_lines.append(
            Line2D([], [], color=cpal[i], marker="o", linestyle="None", markersize=5)
        )

    plt.xlabel(plt_dict["xlabel"])
    plt.ylabel(plt_dict["ylabel"])
    plt.title(plt_dict["title"])

    lgd = plt.legend(
        custom_lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -defaults.DEFAULT_TEXT_SPACING),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    plt.imshow(arr_rgb)

    plt.savefig(file_path, bbox_extra_artists=(lgd,), bbox_inches="tight")
