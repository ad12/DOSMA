from importlib import util

_SUPPORTED_PACKAGES = {}


def _package_available(name):
    """
    Returns:
        bool: Whether module exists in environment.
    """
    global _SUPPORTED_PACKAGES
    if name not in _SUPPORTED_PACKAGES:
        _SUPPORTED_PACKAGES[name] = util.find_spec(name) is not None
    return _SUPPORTED_PACKAGES[name]


def sitk_available():
    return _package_available("SimpleITK")
