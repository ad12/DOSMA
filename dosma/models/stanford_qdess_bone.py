from copy import deepcopy
import time
import numpy as np
import skimage
import gc
from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import SAGITTAL
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel, fill_holes, get_connected_segments, whiten_volume

import SimpleITK as sitk
from tensorflow.keras.models import load_model

__all__ = [
    "StanfordQDessBoneUNet2D", "StanfordQDessBoneUNet2DCoronal", 
    "StanfordQDessBoneUNet2DAxial", "StanfordQDessBoneUNet2DSagittal",
    "StanfordQDessBoneUNet2DSTAPLE"
    ]


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

    TARGET_ORIENTATION = SAGITTAL
    DEFAULT_IMAGE_SIZE = (512, 512)

    def __init__(
        self,
        model_path: str,
        resample_images: bool = True,
        orig_model_image_size: tuple = None,
        tissue_names: tuple = ("pc", "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib", "pat"),
        tissues_to_combine: tuple = (
            (("lat_men", "med_men"), "men"),
            (("mtc", "ltc"), "tc"),
        ),
        bone_indices: tuple = (7, 8, 9)
    ):
        """
        Args:
            model_path (str): Path to model & weights file.
            resample_images (bool): Whether or not to resample input volumes to
                match original model size. If False, will build new model specific
                to loaded image and will load model weights only. Default: True.
        """

        if orig_model_image_size is None:
            orig_model_image_size = self.DEFAULT_IMAGE_SIZE

        self.batch_size = preferences.segmentation_batch_size
        self.orig_model_image_size = orig_model_image_size
        self.resample_images = resample_images
        self.seg_model = self.build_model(model_path=model_path)

        self.tissue_names = tissue_names
        self.tissues_to_combine = tissues_to_combine
        self.bone_indices = bone_indices

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
        fill_bone_holes: bool = True,
    ):
        """Segment tissues.

        Args:
            volume (MedicalVolume): The volume to segment. Either 3D or 4D.
                If the volume is 3D, it is assumed to be the root-sum-of-squares (RSS)
                of the two qDESS echoes. If 4D, volume must be of the shape ``(..., 2)``,
                where the last dimension corresponds to echo 1 and 2, respectively.
            connected_only (bool): If True, only the largest connected component of
                each tissue is returned. Default: True.
            fill_bone_holes (bool): If True, fill holes in bone segmentations. Default: True.

        Returns:
            dict: Dictionary of segmented tissues.
        """
        ndim = volume.ndim
        if ndim not in (3, 4):
            raise ValueError("`volume` must either be 3D or 4D")

        vol_copy = deepcopy(volume)

        if ndim == 4:
            # if 4D, assume last dimension is echo 1 and 2
            vol_copy = np.sqrt(np.sum(vol_copy ** 2, axis=-1))

        # reorient to the sagittal plane
        vol_copy.reformat(self.TARGET_ORIENTATION, inplace=True)

        vol = vol_copy.volume
        vol = self.__preprocess_volume__(vol)

        # reshape volumes to be (slice, 1, x, y)
        v = np.transpose(vol, (2, 0, 1))
        v = np.expand_dims(v, axis=1)

        mask = self.seg_model.predict(v, batch_size=self.batch_size, verbose=1)

        # return mask
        # one-hot encode mask, reorder axes, and re-size to input shape
        mask = self.__postprocess_segmentation__(
            mask, connected_only=connected_only, fill_bone_holes=fill_bone_holes
        )

        # Create temporary dictionary to hold target-oriented volumes
        vols_target = {}

        # Create 'all' volume in target orientation
        vol_all_target = deepcopy(vol_copy)
        vol_all_target.volume = deepcopy(mask)
        vols_target["all"] = vol_all_target

        # Create individual tissues in target orientation
        for i, category in enumerate(self.tissue_names):
            vol_target = deepcopy(vol_copy)
            vol_target.volume = np.zeros_like(mask)
            vol_target.volume[mask == i + 1] = 1
            vols_target[category] = vol_target

        # Combine tissues in target orientation space
        for tissues, tissue_name in self.tissues_to_combine:
            vol_target = deepcopy(vol_copy)
            vol_target.volume = np.zeros_like(mask)
            # Use logical OR instead of addition for boolean arrays
            vol_target.volume[(vols_target[tissues[0]].volume == 1) | (vols_target[tissues[1]].volume == 1)] = 1
            vols_target[tissue_name] = vol_target

        # Now reformat all volumes to original orientation
        vols = {}
        for name, vol_target in vols_target.items():
            vol_cp = deepcopy(vol_target)
            vol_cp.reformat(volume.orientation, inplace=True)
            vols[name] = vol_cp

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

    def __postprocess_segmentation__(
        self, mask: np.ndarray, connected_only: bool = True, fill_bone_holes: bool = True
    ):

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

        if fill_bone_holes is True:
            for bone_idx in self.bone_indices:
                mask_ = fill_holes(mask, label_idx=bone_idx)
                mask[mask_ == 1] = bone_idx

        mask = mask.astype(np.uint8)

        return mask

class StanfordQDessBoneUNet2DCoronal(StanfordQDessBoneUNet2D):
    """2D UNet for bone segmentation in coronal plane"""
    ALIASES = ("stanford-qdess-2022-unet2d-bone-coronal",)
    CORONAL_TRANSPOSED  = ('LR', 'SI', 'AP')
    TARGET_ORIENTATION = CORONAL_TRANSPOSED
    DEFAULT_IMAGE_SIZE = (160, 512)  


class StanfordQDessBoneUNet2DAxial(StanfordQDessBoneUNet2D):
    """2D UNet for bone segmentation in axial plane"""
    ALIASES = ("stanford-qdess-2022-unet2d-bone-axial",)
    AXIAL_TRANSPOSED = ('LR', 'AP', 'SI')
    TARGET_ORIENTATION = AXIAL_TRANSPOSED
    DEFAULT_IMAGE_SIZE = (160, 512)  # Example different size

class StanfordQDessBoneUNet2DSagittal(StanfordQDessBoneUNet2D):
    """2D UNet for bone segmentation in sagittal plane"""
    ALIASES = ("stanford-qdess-2022-unet2d-bone-sagittal",)
    TARGET_ORIENTATION = SAGITTAL
    DEFAULT_IMAGE_SIZE = (512, 512)  # Example different size


class StanfordQDessBoneUNet2DSTAPLE():
    """
    This model applies the sagittal, coronal, and axial UNet
    models to the input volume and then combines the results
    using STAPLE.
    """
    
    # need to combine labels from multiple models - but only trust
    # some models for certain tissues. 
    # ("pc", "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib", "pat")
    # Sag - include all of them. 
    # Cor - only: "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib",
    # Ax - only: "pc", "fem", "tib", "pat"
    
    list_idx_not_include_STAPLE = [
        [], # what not to include for sagittal
        [1], # what not to include for coronal
        [2, 3, 4, 5, 6] # what not to include for axial
    ]
    # dict_tissues_combine_staple = {
    #     "pc": ["sag", "ax"],
    #     "fc": ["sag", "cor", "ax"],
    #     "mtc": ["sag", "cor"],
    #     "ltc": ["sag", "cor"],
    #     "med_men": ["sag", "cor"],
    #     "lat_men": ["sag", "cor"],
    #     "fem": ["sag", "cor", "ax"],
    #     "tib": ["sag", "cor", "ax"],
    #     "pat": ["sag", "ax"]
    # }
    dict_plane_idx = {
        "sag": 0,
        "cor": 1,
        "ax": 2
    }

    def __init__(
        self, 
        sagittal_model_path, coronal_model_path, axial_model_path,
        tissue_names: tuple = ("pc", "fc", "mtc", "ltc", "med_men", "lat_men", "fem", "tib", "pat"),
        tissues_to_combine: tuple = (
            (("lat_men", "med_men"), "men"),
            (("mtc", "ltc"), "tc"),
        ),
        verbose=False,
    ):
        
        self.sagittal_model_path = sagittal_model_path
        self.coronal_model_path = coronal_model_path
        self.axial_model_path = axial_model_path
        self.tissue_names = tissue_names
        self.tissues_to_combine = tissues_to_combine
        self.verbose = verbose
    def generate_mask(self, volume: MedicalVolume):
        """
        iterate over the models, loading them, generating masks, 
        then deleting them from memory - don't want to have 
        a GPU memory issue. 
        """
        vol_copy = deepcopy(volume)
        
        list_models = [
            [self.sagittal_model_path, StanfordQDessBoneUNet2DSagittal],
            [self.coronal_model_path, StanfordQDessBoneUNet2DCoronal],
            [self.axial_model_path, StanfordQDessBoneUNet2DAxial]
        ]

        masks = []
        for model_idx, (model_path, model_class) in enumerate(list_models):
            start_time = time.time()
            model = model_class(model_path)
            masks_dict_ = model.generate_mask(volume)
            masks.append(masks_dict_["all"])
            del model
            gc.collect()
            if self.verbose:
                print(f"Time taken to generate mask {model_idx}: {time.time() - start_time} seconds")
        
        # for each mask, go in and set the regions we are not using to zero. 
        tic = time.time()
        for i, mask in enumerate(masks):
            for idx in self.list_idx_not_include_STAPLE[i]:
                mask.volume[mask.volume == idx] = 0
        if self.verbose:
            print(f"Time taken to set the regions we are not using to zero: {time.time() - tic} seconds")
        tic = time.time()
        masks_sitk = [mask.to_sitk() for mask in masks]
        # cast to uint16 to prevent overflow
        masks_sitk = [sitk.Cast(mask, sitk.sitkUInt16) for mask in masks_sitk]
        
        # unpack the sitk_masks
        staple_mask_sitk = sitk.MultiLabelSTAPLE(*masks_sitk)
        
        if self.verbose:
            print(f"Time to run STAPLE: {time.time() - tic} seconds")
        
        tic = time.time()
        
        staple_mask_mv = MedicalVolume.from_sitk(staple_mask_sitk)
        # set any labels > 9 (the expected max label) to 0
        # these were likely undecided labels in the STAPLE algorithm
        staple_mask_mv.volume[staple_mask_mv.volume > 9] = 0
        # set to uint8
        staple_mask_mv.volume = staple_mask_mv.volume.astype(np.uint8)
        staple_mask_mv.reformat(volume.orientation, inplace=True)
        
        # now... create the individual tissue masks as was expected/previously done by
        # the other models. 
        # Create temporary dictionary to hold target-oriented volumes
        vols = {}
        # Create 'all' volume in target orientation
        vols["all"] = staple_mask_mv

        # Create individual tissues in target orientation
        for i, category in enumerate(self.tissue_names):
            vol = deepcopy(vol_copy)
            vol.volume = np.zeros_like(staple_mask_mv.volume)
            vol.volume[staple_mask_mv.volume == i + 1] = 1
            vols[category] = vol

        # Combine tissues in target orientation space
        for tissues, tissue_name in self.tissues_to_combine:
            vol = deepcopy(vol_copy)
            vol.volume = np.zeros_like(staple_mask_mv.volume)
            # Use logical OR instead of addition for boolean arrays
            vol.volume[(vols[tissues[0]].volume == 1) | (vols[tissues[1]].volume == 1)] = 1
            vols[tissue_name] = vol

        if self.verbose:
            print(f"Time taken to create the individual tissue masks: {time.time() - tic} seconds")

        return vols
        

    def __combine_masks__(self, list_masks, vol_copy):
        """
        Combine the masks from the different planes.
        """
        
        # this should use STAPLE algorithm
        # this is build into SimpleITK
        # need to convert MedicalVolume to SimpleITK image
        # then do combination
        # then convert back to MedicalVolume
        # Then return the MedicalVolume
