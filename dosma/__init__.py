"""The core module contains functions and classes for medical image analysis.
"""
from dosma.utils.logger import setup_logger  # noqa

from dosma import core as _core  # noqa: E402

from dosma.core import *  # noqa
from dosma.defaults import preferences  # noqa
from dosma.utils.collect_env import collect_env_info  # noqa
from dosma.utils.env import debug  # noqa

from dosma.core.med_volume import MedicalVolume  # noqa: F401
from dosma.core.io.format_io_utils import read, write  # noqa: F401
from dosma.core.io.format_io import ImageDataFormat  # noqa: F401
from dosma.core.io.dicom_io import DicomReader, DicomWriter  # noqa: F401
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter  # noqa: F401
from dosma.core.device import Device, get_device, to_device  # noqa: F401
from dosma.core.orientation import to_affine  # noqa: F401
from dosma.core.registration import (  # noqa: F401
    register,
    apply_warp,
    symlink_elastix,
    unlink_elastix,
)
from dosma.core.fitting import (  # noqa: F401
    CurveFitter,
    PolyFitter,
    MonoExponentialFit,
    curve_fit,
    polyfit,
)

import dosma.core.numpy_routines as numpy_routines  # noqa: F401


__all__ = []
__all__.extend(_core.__all__)

setup_logger()

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.1.2"
