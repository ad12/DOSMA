import keras.backend as K
import numpy as np
from keras.layers import BatchNormalization as BN
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate
from keras.models import Model

from models.model import SegModel


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

    def generate_mask(self, volume):
        if type(volume) is not np.ndarray or len(volume.shape) != 3:
            raise ValueError("Volume must be a 3D numpy array")

        # reshape volume to be (slice, x, y, 1)
        v = np.transpose(volume, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)
        model = self.keras_model

        mask = model.predict(v, batch_size=self.batch_size)
        mask = (mask > self.sigmoid_threshold).astype(np.uint8)

        # reshape mask to be (x, y, slice)
        mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

        K.clear_session()
        return mask
