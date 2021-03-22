from importlib import util
from typing import Tuple

_SUPPORTED_PACKAGES = {}


def package_available(name: str):
    """Returns if package is available.

    Args:
        name (str): Name of the package.

    Returns:
        bool: Whether module exists in environment.
    """
    global _SUPPORTED_PACKAGES
    if name not in _SUPPORTED_PACKAGES:
        _SUPPORTED_PACKAGES[name] = util.find_spec(name) is not None
    return _SUPPORTED_PACKAGES[name]


def get_version(package_or_name, num: int = 2) -> Tuple[int]:
    """Returns version number of the package up to ``num``.

    Args:
        package_or_name (``module`` or ``str``): Module or name of module.
            This package must have the version accessible through ``<module>.__version__``.
        num (int, optional): Version depth to return. Should be <=3 following standard
            semantic versioning. For example, ``num==2`` will return ``(X, Y)``.

    Returns:
        Tuple[int]: The versions.

    Examples:
        >>> get_version("numpy")
        (1, 20)
        >>> get_version("numpy", 3)
        (1, 20, 1)
        >>> get_version("numpy") >= (1, 18)
        True
    """
    if isinstance(package_or_name, "str"):
        if not package_available(package_or_name):
            raise ValueError(f"Package {package_or_name} not available")
        package_or_name = util.find_spec(package_or_name)
    version = package_or_name.__version__
    return tuple([int(x) for x in version.split(".")[:num]])


def sitk_available():
    return package_available("SimpleITK")


def cupy_available():
    return package_available("cupy")


def sigpy_available():
    return package_available("sigpy")


def torch_available():
    return package_available("torch")
