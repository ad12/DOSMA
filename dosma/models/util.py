"""
Utility functions common to multiple models and their files.
"""

import tensorflow as tf


def get_tensor_shape_as_list(x):
    """
    Get the shape of a tensor as a list

    Args:
        x: tf.Tensor or tf.TensorShape or list or tuple

    Returns:
        list: shape of the tensor

    Notes: This was implemented becuase getting conv.shape was returning a tuple in some
    versions of tensorflow/keras.

    """

    # # handle different tensorflow/keras versions returning tuple vs. tf.TensorShape
    shape_ = x.shape
    if isinstance(shape_, list):
        return shape_
    elif isinstance(shape_, tuple):
        return list(shape_)
    elif isinstance(shape_, tf.TensorShape):
        return shape_.as_list()
