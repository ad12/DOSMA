from dosma.models.seg_model import SegModel, whiten_volume
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.core.orientation import SAGITTAL

from keras.models import load_model
from typing import Tuple


from copy import deepcopy

import numpy as np
import skimage

__all__ = ["StanfordQDessBoneUNet2D"]

class StanfordQDessBoneUNet2D(SegModel):
    """
    Template for 2D U-Net models trained on the SKM-TEA dataset (previously
    *2021 Stanford qDESS Knee* dataset).

    This model segments patellar cartilage ("pc"), femoral cartilage ("fc"),
    tibial cartilage ("tc"), and the meniscus ("men") from quantitative
    double echo steady state (qDESS) knee scans. The segmentation is computed on
    the root-sum-of-squares (RSS) of the two echoes.

    There are a few weights files that are associated with this model.
    We provide a short description of each below:

        *   ``qDESS_2021_v1-rms-unet2d-pc_fc_tc_men_weights.h5``: This is the baseline
            model trained on the SKM-TEA dataset (v1.0.0).
        *   ``qDESS_2021_v0_0_1-rms-pc_fc_tc_men_weights.h5``: This model is trained on the RSS
            2021 Stanford qDESS knee dataset (v0.0.1).
        *   ``qDESS_2021_v0_0_1-traintest-rms-pc_fc_tc_men_weights.h5``: This model
            is trained on both the train and test set of the 2021 Stanford qDESS knee
            dataset (v0.0.1).

    Examples:

        >>> # Create model based on the volume's shape (SI, AP, 1).
        >>> model = StanfordQDessUNet2D((256, 256, 1), "/path/to/weights")

        >>> # Generate mask from root-sum-of-squares (rss) volume.
        >>> model.generate_mask(rss)

        >>> # Generate mask from dual-echo volume `de_vol` - shape: (SI, AP, LR, 2)
        >>> model.generate_mask(de_vol)
    """

    ALIASES = ("stanford-qdess-2022-unet2d-bone", "skm-tea-unet2d-bone")

    def __init__(
        self,
        model_path: str,
        image_size: Tuple[int, int] = (384, 384),
        # *args, 
        # **kwargs
    ):
        self.batch_size = preferences.segmentation_batch_size
        self.image_size = image_size
        self.seg_model = self.build_model(model_path=model_path)
        # super().__init__(*args, **kwargs)
    
        

    def build_model(self, model_path: str):
        """
        Loads a segmentation model and its weights.

        Args:
            weights_path:

        Returns:
            Keras segmentation model
        """

        model = load_model(model_path, compile=False)

        return model

    def generate_mask(
        self, 
        volume: MedicalVolume,        
    ):
        """Segment tissues.

        Args:
            volume (MedicalVolume): The volume to segment. Either 3D or 4D.
                If the volume is 3D, it is assumed to be the root-sum-of-squares (RSS)
                of the two qDESS echoes. If 4D, volume must be of the shape ``(..., 2)``,
                where the last dimension corresponds to echo 1 and 2, respectively.
        """
        ndim = volume.ndim
        if ndim not in (3, 4):
            raise ValueError("`volume` must either be 3D or 4D")

        vol_copy = deepcopy(volume)

        if ndim == 4:
            # if 4D, assume last dimension is echo 1 and 2
            vol_copy = np.sqrt(np.sum(vol_copy ** 2, axis=-1))

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
        mask = self.__postprocess_segmentation__(mask)
        
        vol_cp = deepcopy(vol_copy)
        vol_cp.volume = deepcopy(mask)
        # reorient to match with original volume
        vol_cp.reformat(volume.orientation, inplace=True)
        vols = {
            "all": vol_cp
        }

        for i, category in enumerate(["pc", "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib", "pat"]):
            vol_cp = deepcopy(vol_copy)
            vol_cp.volume = np.zeros_like(mask)
            vol_cp.volume[mask == i+1] = 1

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

        volume = skimage.transform.resize(
            image=volume,
            output_shape=self.image_size + (volume.shape[-1], ),
            order=3
        )

        return whiten_volume(volume, eps=1e-8)

    def __postprocess_segmentation__(self, mask: np.ndarray):
        
        # USE ARGMAX TO GET SINGLE VOLUME SEGMENTATION OF ALL TISSUES
        mask = np.argmax(mask, axis=1)
        # # reshape mask to be (x, y, slice)
        mask = np.transpose(mask, (1, 2, 0))

        mask = skimage.transform.resize(
            image=mask,
            output_shape=self.original_image_shape,
            order=0
        )

        return mask