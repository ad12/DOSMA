from models.unet2d import Unet2D

# Network architectures currently supported
SUPPORTED_MODELS = ['unet2d']


def get_model(model_str, input_shape, weights_path):
    """Get a Keras model
    :param model_str: model identifier
    :param input_shape: tuple or list of tuples for initializing input(s) into Keras model
    :param weights_path: filepath to weights used to initialize Keras model
    :return: a Keras model
    """
    if model_str == 'unet2d':
        return Unet2D(input_shape, weights_path)

    raise ValueError('%s model type not supported' % model_str)
