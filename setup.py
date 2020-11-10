#!/usr/bin/env python

import os
from os import path
from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "dosma", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        l.strip() for l in init_py if l.startswith("__version__")  # noqa: E741
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("DOSMA_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [
            l for l in init_py if not l.startswith("__version__")  # noqa: E741
        ]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="dosma",
    version=get_version(),
    author="Arjun Desai, et al.",
    url="https://ad12.github.io/DOSMA",
    description="An AI-powered open-source toolbox for medical image analysis",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "configparser",
        "cython",
        "dicom2nifti",
        "h5py",
        "natsort",
        "nested-lookup",
        "nibabel",
        "nipy",
        "nipype",
        "opencv-python",
        "pandas",
        "pydicom",
        "scikit-image",
        "scipy",
        "seaborn",
        "openpyxl",
        "Pmw",
    ]
)
