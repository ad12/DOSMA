"""The core module contains functions and classes for medical image analysis.
"""
from dosma.utils.logger import setup_logger  # noqa

setup_logger()

from dosma import core as _core  # noqa: E402

from dosma.core import *  # noqa
from dosma.defaults import preferences  # noqa
from dosma.utils.collect_env import collect_env_info  # noqa
from dosma.utils.env import debug  # noqa

__all__ = []
__all__.extend(_core.__all__)

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.1.0"
