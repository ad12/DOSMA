from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import binary_fill_holes

from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.tissues.tissue import largest_cc

try:
    import tensorflow.keras.backend as K
except ImportError:  # pragma: no cover
    pass


class SegModel(ABC):
    """
    Args:
        input_shape (Tuple[int] or List[Tuple[int]]): Shapes(s) for initializing input(s)
            into model in format (height, width, channels).
        weights_path (str): filepath to weights used to initialize Keras model
        force_weights (`bool`, optional): If `True`, load weights without any checking.
            Keras/Tensorflow only.
    """

    ALIASES = [""]  # each segmentation model must have an alias

    def __init__(self, input_shape, weights_path, force_weights=False):
        self.batch_size = preferences.segmentation_batch_size
        self.seg_model = self.build_model(input_shape, weights_path)

    @abstractmethod
    def build_model(self, input_shape, weights_path):
        """
        Builds a segmentation model architecture and loads weights.

        Args:
            input_shape: Input shape of volume
            weights_path:

        Returns:
            a segmentation model that can be used for segmenting tissues (a Keras/TF/PyTorch model)
        """
        pass

    @abstractmethod
    def generate_mask(self, volume: MedicalVolume):
        """Segment the MRI volumes.

        Args:
            volume (MedicalVolume): Volume to segment in proper orientation.

        Returns:
            :class:`MedicalVolume` or List[MedicalVolume]: Volumes are binarized (0,1)
                uint8 3D ndarray of shape ``volume.shape``.

        Raises:
            ValueError: If volumes is not 3D ndarray
                or if tissue is not a string or not in list permitted tissues.

        """
        pass

    def __call__(self, *args, **kwargs):
        return self.generate_mask(*args, **kwargs)

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

    def build_model(self, input_shape, weights_path=None):
        keras_model = self.__load_keras_model__(input_shape)
        if weights_path:
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

    def __del__(self):
        K.clear_session()


# ============================ Preprocessing utils ============================
__VOLUME_DIMENSIONS__ = 3
__EPSILON__ = 1e-8


def whiten_volume(x: np.ndarray, eps: float = 0.0):
    """Whiten volumes by mean and std of all pixels.

    Args:
        x (ndarray): 3D numpy array (MRI volumes)
        eps (float, optional): Epsilon to avoid division by 0.

    Returns:
        ndarray: A numpy array with mean ~ 0 and standard deviation ~ 1
    """
    if len(x.shape) != __VOLUME_DIMENSIONS__:
        raise ValueError(f"Input has {x.ndims} dimensions. Expected {__VOLUME_DIMENSIONS__}")

    return (x - np.mean(x)) / (np.std(x) + eps)


# ============================ Postprocessing utils ============================
def get_connected_segments(mask: np.ndarray) -> np.ndarray:
    """
    Get the connected segments in segmentation mask

    Args:
        mask (np.ndarray): 3D volume of all segmented tissues.

    Returns:
        np.ndarray: 3D volume with only the connected segments.
    """

    unique_tissues = np.unique(mask)

    mask_ = np.zeros_like(mask)

    for idx in unique_tissues:
        if idx == 0:
            continue
        mask_binary = np.zeros_like(mask)
        mask_binary[mask == idx] = 1
        mask_binary_conected = np.asarray(largest_cc(mask_binary), dtype=np.uint8)
        mask_[mask_binary_conected == 1] = idx

    return mask_


def fill_holes(mask: np.ndarray, label_idx=None) -> np.ndarray:
    """
    Fill holes in binary mask.

    Args:
        mask (np.ndarray): 3D volume of all segmented tissues.
        label_idx (int, optional): Label index to fill holes in mask.

    Returns:
        np.ndarray: 3D volume with holes filled.
    """

    if label_idx is None:
        assert len(np.unique(mask) == 2), "Mask should be binary"
        mask_ = mask.copy()
    else:
        mask = (mask == label_idx).copy()

    mask_ = binary_fill_holes(mask)

    return mask_
