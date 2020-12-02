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


def get_resources():
    """Get the resources files for dosma. To be used with `package_data`.

    All files under 'dosma/resources/{elastix,templates}'.
    """
    import pathlib

    files = []
    # Elastix files
    for path in pathlib.Path("dosma/resources/elastix").rglob('*.*'):
        files.append(str(path))
    for path in pathlib.Path("dosma/resources/templates").rglob('*.*'):
        files.append(str(path))
    return [x.split("/", 1)[1] for x in files]


setup(
    name="dosma",
    version=get_version(),
    author="Arjun Desai",
    url="https://ad12.github.io/DOSMA",
    description="An AI-powered open-source medical image analysis toolbox",
    packages=find_packages(exclude=("configs", "tests", "tests.*")),
    package_data={'dosma': get_resources(),},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "h5py",
        "natsort",
        "nested-lookup",
        "nibabel",
        "nipype",
        "pandas",
        # TODO Issue #57: Remove pydicom upper bound (https://github.com/ad12/DOSMA/issues/57)
        "pydicom>=1.6.0,<=2.0.0",
        "scikit-image",
        "scipy",
        "seaborn",
        "openpyxl",
        "Pmw",
        "PyYAML",
        "tqdm>=4.42.0"
    ],
    license="GNU",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "dev": ["flake8", "isort", "black==19.3b0", "sphinx", "sphinxcontrib.bibtex"],
    },
)
