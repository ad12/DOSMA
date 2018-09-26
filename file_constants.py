import os

# Elastix files

__DIR__ = os.path.dirname(__file__)
__PATH_TO_ELASTIX_FOLDER__ = os.path.join(__DIR__, 'elastix_params')

ELASTIX_AFFINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-affine.txt')
ELASTIX_BSPLINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-bspline.txt')
ELASTIX_RIGID_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-rigid.txt')


# Temporary file path
TEMP_FOLDER_PATH = os.path.join(__DIR__, 'temp')

# TODO: nipype logging

# Pixel Spacing keys
