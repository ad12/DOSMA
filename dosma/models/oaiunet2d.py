"""Models trained on the Osteoarthritis Initiative (OAI) dataset(s)."""

import os
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

__all__ = ["OAIUnet2D", "IWOAIOAIUnet2D", "IWOAIOAIUnet2DNormalized"]


class OAIUnet2D(KerasSegModel):
    """Model trained in Chaudhari et al. IWOAI 2018

    Original Github: https://github.com/akshaysc/msk_segmentation
    """

    ALIASES = ["oai-unet2d", "oai_unet2d"]

    sigmoid_threshold = 0.5

    def __load_keras_model__(self, input_shape, force_weights=False):
        """Generate Unet 2D model

        Args:
            input_shape: tuple of input size - format: (height, width, 1)

        Returns:
            A Keras model

        Raises:
            ValueError: If ``input_size`` is not tuple or dimensions
                or ``input_size`` does not match (height, width, 1)
        """
        if not _SUPPORTS_KERAS:
            raise ImportError(
                "`oaiunet2d` segmentation models depend on tensorflow/keras backends. "
                "Install them with `pip install tensorflow; pip install keras`"
            )

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
        recon = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv)

        model = Model(inputs=[inputs], outputs=[recon])

        return model

    def generate_mask(self, volume: MedicalVolume):
        vol_copy = deepcopy(volume)

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

        # reshape mask to be (x, y, slice)
        mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

        vol_copy.volume = mask

        # reorient to match with original volume
        vol_copy.reformat(volume.orientation, inplace=True)

        return vol_copy

    def __preprocess_volume__(self, volume: np.ndarray):
        # TODO: Remove epsilon if difference in performance difference is not large.
        return whiten_volume(volume, eps=1e-8)


class IWOAIOAIUnet2D(OAIUnet2D):
    """
    Model trained by Team 6 in the 2019 IWOAI Segmentation Challenge.

    References:
        Desai, et al., "The International Workshop on Osteoarthritis Imaging Knee
        MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis
        Framework on a Standardized Dataset." arXiv preprint arXiv:2004.14003
        (2020). `[link] <https://arxiv.org/abs/2004.14003>`_
    """

    ALIASES = ["iwoai-2019-t6"]
    _WEIGHTS_FILE = "iwoai-2019-unet2d_fc-tc-pc-men_weights.h5"

    def __init__(self, input_shape, weights_path, force_weights=False):
        if not force_weights and os.path.basename(weights_path) != self._WEIGHTS_FILE:
            raise ValueError(f"Weights {weights_path} not supported for {type(self)}")
        super().__init__(input_shape, weights_path)

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
        vol_copy = deepcopy(volume)

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
        for i, category in enumerate(["fc", "tc", "pc", "men"]):
            vol_cp = deepcopy(vol_copy)
            vol_cp.volume = mask[..., i]

            # reorient to match with original volume
            vol_cp.reformat(volume.orientation, inplace=True)
            vols[category] = vol_cp

        return vols

    def __preprocess_volume__(self, volume: np.ndarray):
        return volume


class IWOAIOAIUnet2DNormalized(IWOAIOAIUnet2D):
    """
    Extension of model trained by Team 6 in the 2019 IWOAI Segmentation
    Challenge (with normalization).

    This model uses the same architecture as :class:`IWOAIOAIUnet2D`,
    but pre-processes the input data by zero-mean, unit-std normalization.

    References:
        Desai, et al., "The International Workshop on Osteoarthritis Imaging Knee
        MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis
        Framework on a Standardized Dataset." arXiv preprint arXiv:2004.14003
        (2020).
    """

    ALIASES = ("iwoai-2019-t6-normalized",)
    _WEIGHTS_FILE = "iwoai-2019-unet2d-normalized_fc-tc-pc-men_weights.h5"

    def __preprocess_volume__(self, volume: np.ndarray):
        return whiten_volume(volume)
