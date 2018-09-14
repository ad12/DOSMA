from abc import ABC, abstractmethod


class SegModel(ABC):
    batch_size = 32

    def __init__(self, input_shape, weights_path):
        self.keras_model = self.__load_keras_model__(input_shape)
        self.keras_model.load_weights(weights_path, by_name=True)

    @abstractmethod
    def __load_keras_model__(self, input_shape):
        pass

    @abstractmethod
    def generate_mask(self, volume):
        pass
