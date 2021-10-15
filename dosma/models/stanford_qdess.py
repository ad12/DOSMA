"""
Models trained on the 2021 Stanford quantitative double echo
steady state (qDESS) knee dataset.
"""


from copy import deepcopy

import numpy as np

try:
    from keras.layers import BatchNormalization as BN
    from keras.layers import Concatenate, Conv2D, Conv2DTranspose, Dropout, Input, MaxPooling2D
    from keras.models import Model

    _SUPPORTS_KERAS = True
except ImportError:  # pragma: no-cover
    _SUPPORTS_KERAS = False  # pragma: no-cover

from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import SAGITTAL
from dosma.models.seg_model import KerasSegModel, whiten_volume

__all__ = ["StanfordQDessUNet2D"]


class StanfordQDessUNet2D(KerasSegModel):
    """
    2D U-Net model trained on the 2021 Stanford qDESS Knee dataset.

    This model segments patellar cartilage ("pc"), femoral cartilage ("fc"),
    tibial cartilage ("tc"), and the meniscus ("men") from quantitative
    double echo steady state (qDESS) knee scans. The segmentation is computed on
    the root-sum-of-squares (RSS) of the two echoes.

    There are a few weights files that are associated with this model.
    We provide a short description of each below:

        * ``qDESS_2021_v1-rms-unet2d-pc_fc_tc_men_weights.h5``: This is the baseline
            model trained on the 2021 Stanford qDESS knee dataset (v1.0.0).
        * ``qDESS_2021_v0_0_1-rms-pc_fc_tc_men_weights.h5``: This model is trained on the RSS
            2021 Stanford qDESS knee dataset (v0.0.1).
        * ``qDESS_2021_v0_0_1-traintest-rms-pc_fc_tc_men_weights.h5``: This model
            is trained on both the train and test set of the 2021 Stanford qDESS knee
            dataset (v0.0.1).

    Examples:
        # Create model based on the volume's shape (SI, AP, 1).
        >>> model = StanfordQDessUNet2D((256, 256, 1), "/path/to/weights")

        # Generate mask from root-sum-of-squares (rss) volume.
        >>> model.generate_mask(rss)

        # Generate mask from dual-echo volume `de_vol` - shape: (SI, AP, LR, 2)
        >>> model.generate_mask(de_vol)
    """

    ALIASES = ("stanford-qdess-2021-unet2d",)

    sigmoid_threshold = 0.5

    def __load_keras_model__(self, input_shape):
        if type(input_shape) is not tuple or len(input_shape) != 3 or input_shape[2] != 1:
            raise ValueError("input_size must be a tuple of size (height, width, 1)")

        nfeatures = [2 ** feat * 32 for feat in np.arange(6)]
        depth = len(nfeatures)

        conv_ptr = []

        # input layer
        inputs = Input(input_shape)

        # step down convolutional layers
        pool = inputs
        for depth_cnt in range(depth):

            conv = Conv2D(
                nfeatures[depth_cnt],
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(pool)
            conv = Conv2D(
                nfeatures[depth_cnt],
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(conv)

            conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
            conv = Dropout(rate=0.0)(conv)

            conv_ptr.append(conv)

            # Only maxpool till penultimate depth
            if depth_cnt < depth - 1:

                # If size of input is odd, only do a 3x3 max pool
                xres = conv.shape.as_list()[1]
                if xres % 2 == 0:
                    pooling_size = (2, 2)
                elif xres % 2 == 1:
                    pooling_size = (3, 3)

                pool = MaxPooling2D(pool_size=pooling_size)(conv)

        # step up convolutional layers
        for depth_cnt in range(depth - 2, -1, -1):

            deconv_shape = conv_ptr[depth_cnt].shape.as_list()
            deconv_shape[0] = None

            # If size of input is odd, then do a 3x3 deconv
            if deconv_shape[1] % 2 == 0:
                unpooling_size = (2, 2)
            elif deconv_shape[1] % 2 == 1:
                unpooling_size = (3, 3)

            up = Concatenate(axis=3)(
                [
                    Conv2DTranspose(
                        nfeatures[depth_cnt], (3, 3), padding="same", strides=unpooling_size
                    )(conv),
                    conv_ptr[depth_cnt],
                ]
            )

            conv = Conv2D(
                nfeatures[depth_cnt],
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(up)
            conv = Conv2D(
                nfeatures[depth_cnt],
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
            )(conv)

            conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
            conv = Dropout(rate=0.00)(conv)

        # combine features
        recon = Conv2D(4, (1, 1), padding="same", activation="sigmoid")(conv)

        model = Model(inputs=[inputs], outputs=[recon])

        return model

    def generate_mask(self, volume: MedicalVolume):
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
            vol_copy = np.sqrt(np.sum(vol_copy ** 2, axis=-1))

        # reorient to the sagittal plane
        vol_copy.reformat(SAGITTAL, inplace=True)

        vol = vol_copy.volume
        vol = self.__preprocess_volume__(vol)

        # reshape volumes to be (slice, x, y, 1)
        v = np.transpose(vol, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)
        model = self.seg_model

        mask = model.predict(v, batch_size=self.batch_size, verbose=1)
        mask = (mask > self.sigmoid_threshold).astype(np.uint8)

        # reshape mask to be (x, y, slice, classes)
        mask = np.transpose(mask, (1, 2, 0, 3))

        vols = {}
        for i, category in enumerate(["pc", "fc", "tc", "men"]):
            vol_cp = deepcopy(vol_copy)
            vol_cp.volume = mask[..., i]

            # reorient to match with original volume
            vol_cp.reformat(volume.orientation, inplace=True)
            vols[category] = vol_cp

        return vols

    def __preprocess_volume__(self, volume: np.ndarray):
        # TODO: Remove epsilon if difference in performance difference is not large.
        return whiten_volume(volume, eps=1e-8)
