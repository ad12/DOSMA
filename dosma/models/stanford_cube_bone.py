from dosma.models.stanford_qdess_bone import StanfordQDessBoneUNet2D

__all__ = ["StanfordCubeBoneUNet2D"]


class StanfordCubeBoneUNet2D(StanfordQDessBoneUNet2D):
    """
    This model segments bones (femur, tibia, patella) from cube knee scans.

    There are a few weights files that are associated with this model.
    We provide a short description of each below:

        *   ````: This is the baseline
            model trained on a subset of the K2S MRI reconstruction challenge
            hosted by UCSF using the 2D network from Gatti et al. MAGMA, 2021.

    By default this class will resample the input to be the size of the trained
    model (512x512) for segmentation and then will re-sample the outputted
    segmentation to match the original volume.

    By default, we return the largest connected component of each tissue. This
    can be disabled by setting `connected_only=False` in the `model.generate_mask()`.

    By default, we fill holes in bone segmentations. This can be disabled by setting
    `fill_bone_holes=False` in the `model.generate_mask()`.

    The output includes individual objects for each segmented tissue. It also
    includes a combined label for all of the tissues in a single 3D mask.

    Examples:

        >>> # Create model.
        >>> model = StanfordCubeBoneUNet2D("/path/to/model.h5")

        >>> # Generate mask from medical volume.
        >>> model.generate_mask(medvol)

        >>> # Generate mask from medical volume without getting largest connected components.
        >>> model.generate_mask(medvol, connected_only=False)

    """

    ALIASES = ("stanford-cube-2024-unet2d-bone", "k2s-unet2d-bone")

    def __init__(
        self,
        model_path: str,
        resample_images: bool = True,
        orig_model_image_size: tuple = (512, 512),
        tissue_names: tuple = ("fem", "tib", "pat"),
        tissues_to_combine: tuple = (),
        bone_indices: tuple = (7, 8, 9)
        # *args,
        # **kwargs
    ):
        super().__init__(
            model_path,
            resample_images,
            orig_model_image_size,
            tissue_names,
            tissues_to_combine,
            bone_indices,
        )
