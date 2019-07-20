"""
Functions for loading Keras models

@author: Arjun Desai
        (C) Stanford University, 2019
"""

from dosma.models.oaiunet2d import OAIUnet2D

__all__ = ['get_model', 'SUPPORTED_MODELS']

# Network architectures currently supported
__SUPPORTED_MODELS__ = [OAIUnet2D]

# Initialize supported models for the command line
SUPPORTED_MODELS = [x.ALIASES[0] for x in __SUPPORTED_MODELS__]


def get_model(model_str, input_shape, weights_path):
    """Get a Keras model
    :param model_str: model identifier
    :param input_shape: tuple or list of tuples for initializing input(s) into Keras model
    :param weights_path: filepath to weights used to initialize Keras model
    :return: a Keras model
    """
    for m in __SUPPORTED_MODELS__:
        if model_str in m.ALIASES:
            return m(input_shape, weights_path)

    raise LookupError('%s model type not supported' % model_str)
