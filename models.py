from __future__ import print_function, division

import numpy as np
import os
import warnings
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Concatenate
from keras.layers import BatchNormalization as BN
import keras.backend as K

# List of tissues that can be segmented
FEMORAL_CARTILAGE_STR = 'fc'
MENISCUS_STR = 'men'
PATELLAR_CARTILAGE_STR = 'pc'
TIBIAL_CARTILAGE_STR = 'tc'

# Absolute directory where this file lives
__ABS_DIR__ = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DICT = {FEMORAL_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_fc_weights--0.8968.h5'),
                MENISCUS_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_men_weights--0.7692.h5'),
                PATELLAR_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_pc_weights--0.6206.h5'),
                TIBIAL_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_tc_weights--0.8625.h5')}

# Input size that is expected
# All inputs must be at least this size
DEFAULT_INPUT_SIZE = (288, 288, 1)

BATCH_SIZE = 32

def unet_2d_model_keras2(input_size):
    """Generate Unet 2D model compatible with Keras 2

    :param input_size: tuple of input size - format: (height, width, 1)

    :rtype: Keras model

    :raise ValueError if input_size is not tuple or dimensions of input_size do not match (height, width, 1)
    """

    if type(input_size) is not tuple or len(input_size) != 3 or input_size[2] != 1:
        raise ValueError('input_size must be a tuple of size (height, width, 1)')

    nfeatures = [2 ** feat * 32 for feat in np.arange(6)]
    depth = len(nfeatures)

    conv_ptr = []

    # input layer
    inputs = Input(input_size)

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


def unet_2d_model(input_size):
    """Generate Unet 2D model

    ========================================================================
    THIS METHOD IS DEPRECATED. USE unet_2d_model_keras2(input_size) INSTEAD
    ========================================================================

    :param input_size: tuple of input size - should be (height, width, 1)

    :rtype: Keras model

    :raise ValueError if input_size is not tuple or dimensions of input_size do not match (height, width, 1)
    """

    warnings.warn("This function is deprecated. Use unet_2d_model_keras2 instead", DeprecationWarning, stacklevel=2)

    if type(input_size) is not tuple or len(input_size) != 3 or input_size[2] != 1:
        raise ValueError('input_size must be a tuple')

    nfeatures = [2**feat*32 for feat in np.arange(6)]
    depth = len(nfeatures)    

    conv_ptr = []

    # input layer
    inputs = Input(input_size)

    # step down convolutional layers 
    pool = inputs
    for depth_cnt in range(depth):

        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(pool)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Only maxpool till penultimate depth
        if depth_cnt < depth-1:

            # If size of input is odd, only do a 3x3 max pool
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2,2)
            elif (xres % 2 == 1):
                pooling_size = (3,3)

            pool = MaxPooling2D(pool_size=pooling_size)(conv)


    # step up convolutional layers
    for depth_cnt in range(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # If size of input is odd, then do a 3x3 deconv  
        if (deconv_shape[1] % 2 == 0):
            unpooling_size = (2,2)
        elif (deconv_shape[1] % 2 == 1):
            unpooling_size = (3,3)

        up = concatenate([Conv2DTranspose(nfeatures[depth_cnt],(3,3),
                          padding='same',
                          strides=unpooling_size,
                          output_shape=deconv_shape)(conv),
                          conv_ptr[depth_cnt]], 
                          axis=3)

        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(up)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    recon = Conv2D(1, (1,1), padding='same', activation='sigmoid')(conv)

    model = Model(inputs=[inputs], outputs=[recon])
    
    return model


def generate_mask(volume, tissue):
    """Segment the given tissue in the MRI volume

    :param volume: 3D numpy array of shape (height, width, slices)
    :param tissue: FEMORAL_CARTILAGE_STR, MENISCUS_STR, PATELLAR_CARTILAGE_STR, or TIBIAL_CARTILAGE_STR

    :rtype: binarized (0,1) uint8 3D numpy array of shape volume.shape

    @:raise ValueError if volume is not 3D numpy array
    @:raise ValueError if tissue is not a string or not in list permitted tissues

    """

    if type(volume) is not np.ndarray or len(volume.shape) != 3:
        raise ValueError("Volume must be a 3D numpy array")
    if type(tissue) is not str or tissue not in [FEMORAL_CARTILAGE_STR,
                                                 MENISCUS_STR,
                                                 PATELLAR_CARTILAGE_STR,
                                                 TIBIAL_CARTILAGE_STR]:
        raise ValueError("Tissue must be \'%s\', \'%s\', \'%s\', or \'%s\'" % (FEMORAL_CARTILAGE_STR,
                                                                                MENISCUS_STR,
                                                                                PATELLAR_CARTILAGE_STR,
                                                                                TIBIAL_CARTILAGE_STR))
    im_shape = volume.shape
    if im_shape[0] < DEFAULT_INPUT_SIZE[0] or im_shape[1] < DEFAULT_INPUT_SIZE[1]:
        raise ValueError("2D slices must at least be size (288, 288, )")

    if im_shape[:2] != DEFAULT_INPUT_SIZE[:2]:
        UserWarning("Network was trained using 2D slices of size (288, 288). \n"
                      "Optimal results will be achieved using these dimensions. \n")

    # reshape volume to be (slice, x, y, 1)
    v = np.transpose(volume, (2, 0, 1))
    v = np.expand_dims(v, axis=-1)

    model = unet_2d_model_keras2((im_shape[0], im_shape[1], 1))
    tissue_weight_path = WEIGHTS_DICT[tissue]
    model.load_weights(tissue_weight_path, by_name=True)

    # TODO: verify that performance on is not affected by batch count
    mask = model.predict(v, batch_size=BATCH_SIZE)
    mask = (mask > 0.5).astype(np.uint8)

    # reshape mask to be (x, y, slice)
    mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

    K.clear_session()
    return mask

