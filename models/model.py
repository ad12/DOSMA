from abc import ABC, abstractmethod

import defaults


class SegModel(ABC):
    """
    Abstract wrapper for Keras model used for semantic segmentation
    """
    batch_size = defaults.DEFAULT_BATCH_SIZE

    def __init__(self, input_shape, weights_path):
        """
        :param input_shape: tuple or list of tuples for initializing input(s) into Keras model
        :param weights_path: filepath to weights used to initialize Keras model
        """
        self.keras_model = self.__load_keras_model__(input_shape)
        self.keras_model.load_weights(weights_path, by_name=True)

    @abstractmethod
    def __load_keras_model__(self, input_shape):
        """Generate a Keras model

        :param input_shape: tuple or list of tuples for initializing input(s) into Keras model

        :rtype: a Keras model
        """
        pass

    @abstractmethod
    def generate_mask(self, volume):
        """Segment the MRI volume

        :param volume: 3D numpy array of shape (height, width, slices)

        :rtype: binarized (0,1) uint8 3D numpy array of shape volume.shape

        :raise ValueError if volume is not 3D numpy array
        :raise ValueError if tissue is not a string or not in list permitted tissues

        """
        pass
