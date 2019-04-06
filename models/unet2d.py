"""
@author: Arjun Desai
        (C) Stanford University, 2019
"""

from copy import deepcopy

import keras.backend as K
import numpy as np
from keras.layers import BatchNormalization as BN
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate
from keras.models import Model

from data_io.med_volume import MedicalVolume
from data_io.orientation import SAGITTAL
from models.model import SegModel
from utils import dicom_utils


class Unet2D(SegModel):
    sigmoid_threshold = 0.5

    def __load_keras_model__(self, input_shape):
        """Generate Unet 2D model

        :param input_shape: tuple of input size - format: (height, width, 1)

        :rtype: Keras model

        :raise ValueError if input_size is not tuple or dimensions of input_size do not match (height, width, 1)
        """

        if type(input_shape) is not tuple or len(input_shape) != 3 or input_shape[2] != 1:
            raise ValueError('input_size must be a tuple of size (height, width, 1)')

        nfeatures = [2 ** feat * 32 for feat in np.arange(6)]
        depth = len(nfeatures)

        conv_ptr = []

        # input layer
        inputs = Input(input_shape)

        # step down convolutional layers
        pool = inputs
        for depth_cnt in range(depth):

            conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(pool)
            conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(conv)

            conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
            conv = Dropout(rate=0.0)(conv)

            conv_ptr.append(conv)

            # Only maxpool till penultimate depth
            if depth_cnt < depth - 1:

                # If size of input is odd, only do a 3x3 max pool
                xres = conv.shape.as_list()[1]
                if (xres % 2 == 0):
                    pooling_size = (2, 2)
                elif (xres % 2 == 1):
                    pooling_size = (3, 3)

                pool = MaxPooling2D(pool_size=pooling_size)(conv)

        # step up convolutional layers
        for depth_cnt in range(depth - 2, -1, -1):

            deconv_shape = conv_ptr[depth_cnt].shape.as_list()
            deconv_shape[0] = None

            # If size of input is odd, then do a 3x3 deconv
            if (deconv_shape[1] % 2 == 0):
                unpooling_size = (2, 2)
            elif (deconv_shape[1] % 2 == 1):
                unpooling_size = (3, 3)

            up = Concatenate(axis=3)([Conv2DTranspose(nfeatures[depth_cnt], (3, 3),
                                                      padding='same',
                                                      strides=unpooling_size)(conv),
                                      conv_ptr[depth_cnt]])

            conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(up)
            conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(conv)

            conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
            conv = Dropout(rate=0.00)(conv)

        # combine features
        recon = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv)

        model = Model(inputs=[inputs], outputs=[recon])

        return model

    def generate_mask(self, volume: MedicalVolume):
        vol_copy = deepcopy(volume)

        # reorient to the sagittal plane
        vol_copy.reformat(SAGITTAL)

        vol = vol_copy.volume
        vol = self.__preprocess_volume__(vol)

        # reshape volumes to be (slice, x, y, 1)
        v = np.transpose(vol, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)
        model = self.keras_model

        mask = model.predict(v, batch_size=self.batch_size)
        mask = (mask > self.sigmoid_threshold).astype(np.uint8)

        # reshape mask to be (x, y, slice)
        mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

        K.clear_session()

        vol_copy.volume = mask

        # reorient to match with original volume
        vol_copy.reformat(volume.orientation)

        return vol_copy

    def __preprocess_volume__(self, volume: np.ndarray):
        return dicom_utils.whiten_volume(volume)
