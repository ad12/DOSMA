import os
import sys

import h5py
import nibabel
import nipype
import numpy as np
import pandas as pd
import pydicom
import scipy as sp
import skimage
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_env_info():
    """Collect environment information for reporting issues.

    Run this function when reporting issues on Github.
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import dosma  # noqa

        data.append(("dosma", dosma.__version__ + " @" + os.path.dirname(dosma.__file__)))
    except ImportError:
        data.append(("dosma", "failed to import"))

    # Required packages
    data.append(("h5py", h5py.__version__))
    data.append(("nibabel", nibabel.__version__))
    data.append(("nipype", nipype.__version__))
    data.append(("pandas", pd.__version__))
    data.append(("pydicom", pydicom.__version__))
    data.append(("scipy", sp.__version__))
    data.append(("skimage", skimage.__version__))

    # Optional packages
    try:
        import cupy

        data.append(("cupy", cupy.__version__))
    except ImportError:
        pass

    try:
        import tensorflow

        data.append(("tensorflow", tensorflow.__version__))
    except ImportError:
        pass

    try:
        import keras

        data.append(("keras", keras.__version__))
    except ImportError:
        pass

    try:
        import SimpleITK as sitk

        data.append(("SimpleITK", sitk.__version__))
    except ImportError:
        pass

    try:
        import sigpy

        data.append(("sigpy", sigpy.__version__))
    except ImportError:
        pass

    env_str = tabulate(data)
    return env_str
