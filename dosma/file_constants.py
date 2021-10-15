"""Basic file constants to be shared with program"""

import os

# Elastix files

__DIR__ = os.path.abspath(os.path.dirname(__file__))
__OUT_DIR__ = os.path.abspath(os.path.expanduser("~/.dosma"))
_DOSMA_ELASTIX_FOLDER = os.path.join(__DIR__, "resources/elastix")
__PATH_TO_ELASTIX_FOLDER__ = os.path.join(__DIR__, "resources/elastix/params")

ELASTIX_AFFINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, "parameters-affine.txt")
ELASTIX_BSPLINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, "parameters-bspline.txt")
ELASTIX_RIGID_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, "parameters-rigid.txt")

ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE = os.path.join(
    __PATH_TO_ELASTIX_FOLDER__, "parameters-affine-interregister.txt"
)
ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE = os.path.join(
    __PATH_TO_ELASTIX_FOLDER__, "parameters-rigid-interregister.txt"
)

MAPSS_ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE = os.path.join(
    __PATH_TO_ELASTIX_FOLDER__, "parameters-affine-interregister.txt"
)
MAPSS_ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE = os.path.join(
    __PATH_TO_ELASTIX_FOLDER__, "parameters-rigid-interregister.txt"
)
