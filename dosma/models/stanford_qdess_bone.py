from copy import deepcopy
from typing import Tuple

import numpy as np
import skimage

from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import SAGITTAL
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel, whiten_volume
from dosma.tissues.tissue import Tissue, largest_cc

from keras.models import load_model

__all__ = ["StanfordQDessBoneUNet2D"]


class StanfordQDessBoneUNet2D(SegModel):
    """
    This model segments patellar cartilage ("pc"), femoral cartilage ("fc"),
    tibial cartilage ("tc", "mtc", "ltc"), the meniscus ("men", "med_men",
    "lat_men"), and bones ("fem", "tib", "pat") from quantitative
    double echo steady state (qDESS) knee scans. The segmentation is computed on
    the root-sum-of-squares (RSS) of the two echoes.

    There are a few weights files that are associated with this model.
    We provide a short description of each below:

        *   ``coarse_2d_sagittal_v1_11-18_Dec-04-2022.h5``: This is the baseline
            model trained on a subset of the SKM-TEA dataset (v1.0.0) with bone
            labels using the 2D network from Gatti et al. MAGMA, 2021.
        * PROVIDE ANOTHER SET OF WEIGHTS USING SAME MODEL AS stanford_qdess

    By default this class will resample the input to be the size of the trained
    model (384x384) for segmentation and then will re-sample the outputted
    segmentation to match the original volume.

    By default, we y return the largest connected component of each tissue. This
    can be disabled by setting `connected_only=False` in the `model.generate_mask()`.

    The output includes individual objects for each segmented tissue, including
    separate medial/lateral segments of the meniscus and tibial cartilage. It also
    includes a combined label for the meniscus and tibial cartilage, and a combined
    label for all of the tissues in a single 3D mask.

    Examples:

        >>> # Create model.
        >>> model = StanfordQDessBoneUNet2D("/path/to/model.h5")

        >>> # Generate mask from root-sum-of-squares (rss) volume.
        >>> model.generate_mask(rss)

        >>> # Generate mask from dual-echo volume `de_vol` - shape: (SI, AP, LR, 2)
        >>> model.generate_mask(de_vol)

        >>> # Generate mask from rss volume without getting largest connected components.
        >>> model.generate_mask(rss, connected_only=False)

    """

    ALIASES = ("stanford-qdess-2022-unet2d-bone", "skm-tea-unet2d-bone")

    def __init__(
        self,
        model_path: str,
        resample_images: bool = True,
        # *args,
        # **kwargs
    ):
        """
        Args:
            model_path (str): Path to model & weights file.
            resample_images (bool): Whether or not to resample input volumes to
                match original model size. If False, will build new model specific
                to loaded image and will load model weights only. Default: True.
        """

        self.batch_size = preferences.segmentation_batch_size
        self.orig_model_image_size = (384, 384)
        self.resample_images = resample_images
        self.seg_model = self.build_model(model_path=model_path)
        # super().__init__(*args, **kwargs)

    def build_model(self, model_path: str):
        """
        Loads a segmentation model and its weights.

        Args:
            model_path: Path to model & its weights.

        Returns:
            Keras segmentation model
        """
        if self.resample_images is True:
            model = load_model(model_path, compile=False)
        else:
            raise Exception("Segmenting without resampling is not supported yet.")

        return model

    def generate_mask(
        self,
        volume: MedicalVolume,
        connected_only: bool = True,
    ):
        """Segment tissues.

        Args:
            volume (MedicalVolume): The volume to segment. Either 3D or 4D.
                If the volume is 3D, it is assumed to be the root-sum-of-squares (RSS)
                of the two qDESS echoes. If 4D, volume must be of the shape ``(..., 2)``,
                where the last dimension corresponds to echo 1 and 2, respectively.
            connected_only (bool): If True, only the largest connected component of
                each tissue is returned. Default: True.

        Returns:
            dict: Dictionary of segmented tissues.
        """
        ndim = volume.ndim
        if ndim not in (3, 4):
            raise ValueError("`volume` must either be 3D or 4D")

        vol_copy = deepcopy(volume)

        if ndim == 4:
            # if 4D, assume last dimension is echo 1 and 2
            vol_copy = np.sqrt(np.sum(vol_copy**2, axis=-1))

        # reorient to the sagittal plane
        vol_copy.reformat(SAGITTAL, inplace=True)

        vol = vol_copy.volume
        vol = self.__preprocess_volume__(vol)

        # reshape volumes to be (slice, 1, x, y)
        v = np.transpose(vol, (2, 0, 1))
        v = np.expand_dims(v, axis=1)

        mask = self.seg_model.predict(v, batch_size=self.batch_size, verbose=1)

        # return mask
        # one-hot encode mask, reorder axes, and re-size to input shape
        mask = self.__postprocess_segmentation__(mask, connected_only=connected_only)

        vol_cp = deepcopy(vol_copy)
        vol_cp.volume = deepcopy(mask)
        # reorient to match with original volume
        vol_cp.reformat(volume.orientation, inplace=True)
        vols = {"all": vol_cp}

        for i, category in enumerate(
            ["pc", "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib", "pat"]
        ):
            vol_cp = deepcopy(vol_copy)
            vol_cp.volume = np.zeros_like(mask)
            vol_cp.volume[mask == i + 1] = 1

            # reorient to match with original volume
            vol_cp.reformat(volume.orientation, inplace=True)
            vols[category] = vol_cp

        # Create "tc", "men" labels from original models
        tissues_to_combine = (
            (("lat_men", "med_men"), "men"),
            (("mtc", "ltc"), "tc"),
        )
        for tissues, tissue_name in tissues_to_combine:
            vol_cp = deepcopy(vol_copy)
            vol_cp.volume = np.zeros_like(mask)
            vol_cp.volume[(vols[tissues[0]].volume == 1) + (vols[tissues[1]].volume == 1)] = 1
            # reorient to match with original volume
            vol_cp.reformat(volume.orientation, inplace=True)
            vols[tissue_name] = vol_cp

        return vols

    def __preprocess_volume__(self, volume: np.ndarray):
        # TODO: Remove epsilon if difference in performance difference is not large.

        self.original_image_shape = volume.shape

        if self.resample_images is True:
            volume = skimage.transform.resize(
                image=volume, output_shape=self.orig_model_image_size + (volume.shape[-1],), order=3
            )
        else:
            raise Exception("Segmenting without resampling is not supported yet.")

        return whiten_volume(volume, eps=1e-8)

    def __postprocess_segmentation__(self, mask: np.ndarray, connected_only: bool = True):

        # USE ARGMAX TO GET SINGLE VOLUME SEGMENTATION OF ALL TISSUES
        mask = np.argmax(mask, axis=1)
        # # reshape mask to be (x, y, slice)
        mask = np.transpose(mask, (1, 2, 0))

        if self.resample_images is True:
            mask = skimage.transform.resize(
                image=mask, output_shape=self.original_image_shape, order=0
            )
        else:
            raise Exception("Segmenting without resampling is not supported yet.")

        if connected_only is True:
            mask = get_connected_segments(mask)

        return mask


def get_connected_segments(mask: np.ndarray) -> np.ndarray:
    """
    Get the connected segments in segmentation mask

    Args:
        mask (np.ndarray): 3D volume of all segmented tissues.

    Returns:
        np.ndarray: 3D volume with only the connected segments.
    """

    unique_tissues = np.unique(mask)

    mask_ = np.zeros_like(mask)

    for idx in unique_tissues:
        if idx == 0:
            continue
        mask_binary = np.zeros_like(mask)
        mask_binary[mask == idx] = 1
        mask_binary_conected = np.asarray(largest_cc(mask_binary), dtype=np.uint8)
        mask_[mask_binary_conected == 1] = idx

    return mask_
