from models.unet2d import Unet2D

SUPPORTED_MODELS = ['unet2d']

def get_model(model_str, input_shape, weights_path):
    if model_str == 'unet2d':
        return Unet2D(input_shape, weights_path)

    raise ValueError('%s model type not supported' % model_str)