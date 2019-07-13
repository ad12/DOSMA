"""
Abstract classes to wrap Keras model

SegModel: Abstract wrapper for Keras model used for semantic segmentation

@author: Arjun Desai
        (C) Stanford University, 2019
"""

from abc import ABC, abstractmethod

import numpy as np

from data_io.med_volume import MedicalVolume
from defaults import preferences


class SegModel(ABC):
    ALIASES = ['']  # each segmentation model must have an alias

    batch_size = preferences.segmentation_batch_size

    def __init__(self, input_shape, weights_path):
        """
        :param input_shape: tuple or list of tuples for initializing input(s) into model in format (height, width, channels)
        :param weights_path: filepath to weights used to initialize Keras model
        """
        self.seg_model = self.build_model(input_shape, weights_path)

    @abstractmethod
    def build_model(self, input_shape, weights_path):
        """
        Builds a segmentation model architecture and loads weights

        :param input_shape: Input shape of volume
        :param weights_path:
        :return: a segmentation model that can be used for segmenting tissues (a Keras/TF/PyTorch model)
        """
        pass

    @abstractmethod
    def generate_mask(self, volume: MedicalVolume):
        """Segment the MRI volumes

        :param volume: A Medical Volume (height, width, slices)

        :return: A Medical volume or list of Medical Volumes with volume as binarized (0,1) uint8 3D numpy array of shape volumes.shape

        :raise ValueError if volumes is not 3D numpy array
        :raise ValueError if tissue is not a string or not in list permitted tissues

        """
        pass

    def __preprocess_volume__(self, volume: np.ndarray):
        """
        Preprocess volume prior to putting as input into segmentation network
        :param volume: a numpy array
        :return: a preprocessed numpy array
        """
        return volume

    def __postprocess_volume__(self, volume: np.ndarray):
        """
        Post-process logits (probabilities) or binarized mask
        :param volume: a numpy array
        :return: a postprocessed numpy array
        """
        return volume


class KerasSegModel(SegModel):
    """
    Abstract wrapper for Keras model used for semantic segmentation
    """

    def build_model(self, input_shape, weights_path):
        keras_model = self.__load_keras_model__(input_shape)
        keras_model.load_weights(weights_path)

        return keras_model

    @abstractmethod
    def __load_keras_model__(self, input_shape):
        """
        Build Keras architecture

        :param input_shape: tuple or list of tuples for initializing input(s) into Keras model

        :return: a Keras model
        """
        pass


# ============================ Preprocessing utils ============================
__VOLUME_DIMENSIONS__ = 3
__EPSILON__ = 1e-8


def whiten_volume(x):
    """Whiten volumes by mean and std of all pixels
    :param x: 3D numpy array (MRI volumes)
    :rtype: whitened 3D numpy array
    """
    if len(x.shape) != __VOLUME_DIMENSIONS__:
        raise ValueError("Dimension Error: input has %d dimensions. Expected %d" % (x.ndims, __VOLUME_DIMENSIONS__))

    # Add epsilon to avoid dividing by 0
    return (x - np.mean(x)) / (np.std(x) + __EPSILON__)
