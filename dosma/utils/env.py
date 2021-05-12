import os
from importlib import util

_SUPPORTED_PACKAGES = {}

_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

__all__ = ["debug", "get_version", "package_available"]


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


def get_version(package_or_name) -> str:
    """Returns package version.

    Args:
        package_or_name (``module`` or ``str``): Module or name of module.
            This package must have the version accessible through ``<module>.__version__``.

    Returns:
        str: The package version.

    Examples:
        >>> get_version("numpy")
        "1.20.0"
    """
    if isinstance(package_or_name, str):
        if not package_available(package_or_name):
            raise ValueError(f"Package {package_or_name} not available")
        spec = util.find_spec(package_or_name)
        package_or_name = util.module_from_spec(spec)
        spec.loader.exec_module(package_or_name)
    version = package_or_name.__version__
    return version


def debug(value: bool = None) -> bool:
    """Return (and optionally set) debug mode.

    Args:
        value (bool, optional): If specified, sets the debug status.
            If not specified, debug mode is not set, only returned.

    Returns:
        bool: If ``True``, debug mode is active.

    Raises:
        ValueError: If ``value`` is not a supported value.

    Examples:
        >>> debug()  # get debug status, defaults to False
        False
        >>> debug(True)
        True
        >>> debug()
        True
    """
    from dosma.defaults import preferences

    def _is_debug():
        return os.environ.get("DOSMA_DEBUG", "") in ["True", "true"]

    def _toggle_debug(_old_value, _new_value):
        # TODO: Toggle dosma logging to debug mode.
        if _old_value == _new_value:
            return
        if _new_value:
            preferences.set("nipype", value="stream", prefix="logging")
        else:
            preferences.set("nipype", value="file_stderr", prefix="logging")

    if value is not None:
        old_value = _is_debug()
        if isinstance(value, bool):
            os.environ["DOSMA_DEBUG"] = str(value)
        elif isinstance(value, str) and value.lower() in ("true", "false", ""):
            os.environ["DOSMA_DEBUG"] = value
        else:
            raise ValueError(f"Unknown value for debug: '{value}'")

        _toggle_debug(old_value, _is_debug())

    return _is_debug()


def sitk_available():
    return package_available("SimpleITK")


def cupy_available():
    return package_available("cupy")


def sigpy_available():
    return package_available("sigpy")


def torch_available():
    return package_available("torch")


def resources_dir() -> str:
    return os.path.abspath(os.path.join(_FILE_DIRECTORY, "../resources"))


def output_dir() -> str:
    return os.path.abspath(os.path.join(_FILE_DIRECTORY, "../../.dosma"))


def temp_dir() -> str:
    return os.path.join(output_dir(), "temp")


def log_file_path() -> str:
    return os.path.join(output_dir(), "dosma.log")
