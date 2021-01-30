"""
Functions for loading Keras models

@author: Arjun Desai
        (C) Stanford University, 2019
"""
import os
import yaml
from functools import partial
from typing import Sequence

from dosma.models.oaiunet2d import IWOAIOAIUnet2D, IWOAIOAIUnet2DNormalized, OAIUnet2D
from dosma.models.seg_model import SegModel

__all__ = ["get_model", "SUPPORTED_MODELS"]

# Network architectures currently supported
__SUPPORTED_MODELS__ = [OAIUnet2D, IWOAIOAIUnet2D, IWOAIOAIUnet2DNormalized]

# Initialize supported models for the command line
SUPPORTED_MODELS = [x.ALIASES[0] for x in __SUPPORTED_MODELS__]


def get_model(model_str, input_shape, weights_path, **kwargs):
    """Get a Keras model
    :param model_str: model identifier
    :param input_shape: tuple or list of tuples for initializing input(s) into Keras model
    :param weights_path: filepath to weights used to initialize Keras model
    :return: a Keras model
    """
    for m in __SUPPORTED_MODELS__:
        if model_str in m.ALIASES or model_str == m.__name__:
            return m(input_shape, weights_path, **kwargs)

    raise LookupError("%s model type not supported" % model_str)


def model_from_config(cfg_file_or_dict, weights_dir=None, **kwargs) -> SegModel:
    """Builds a new model from a config file.

    This function is useful for building models that have similar structure/architecture
    to existing models supported in DOSMA, but have different weights and categories.
    The config defines what dosma model should be used as a base, what weights should be loaded,
    and what are the categories.

    The config file should be a yaml file that has the following keys:
        * "DOSMA_MODEL": The base model that exists in DOSMA off of which data should be built.
        * "CATEGORIES": The categories that are supposed to be loaded.
        * "WEIGHTS_FILE": The basename of (or full path to) weights that should be loaded.

    Args:
        cfg_file_or_dict (str or dict): The yaml file or dictionary corresponding to the config.
        weights_dir (str): The directory where weights are stored. If not specified, assumes
            "WEIGHTS_FILE" field in the config is the full path to the weights.
        **kwargs: Keyword arguments for base model `__init__`

    Returns:
        SegModel: A segmentation model with appropriate changes to `generate_mask` to produce
            the right masks.
    """

    def _gen_mask(func, *_args, **_kwargs):
        out = func(*_args, **_kwargs)
        if isinstance(out, dict):
            # Assumes that the dict is ordered, which it is for python>=3.6
            out = out.values()
        elif not isinstance(out, Sequence):
            out = [out]
        if not len(categories) == len(out):
            raise ValueError("Got {} outputs, but {} categories".format(len(out), len(categories)))
        return {cat: out for cat, out in zip(categories, out)}

    if isinstance(cfg_file_or_dict, str):
        with open(cfg_file_or_dict, "r") as f:
            cfg = yaml.load(f)
    else:
        cfg = cfg_file_or_dict

    base_model = cfg["DOSMA_MODEL"]
    categories = cfg["CATEGORIES"]
    weights = cfg["WEIGHTS_FILE"]
    if not os.path.isfile(weights):
        assert weights_dir, "`weights_dir` must be specified"
        weights = os.path.join(weights_dir, cfg["WEIGHTS_FILE"])

    try:
        model: SegModel = get_model(base_model, weights_path=weights, force_weights=True, **kwargs)
    except LookupError as e:
        raise LookupError("BASE_MODEL '{}' not supported \n{}".format(base_model, e))

    # Override generate mask function
    model.generate_mask = partial(_gen_mask, model.generate_mask)

    return model
