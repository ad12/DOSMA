"""I/O formatting templates.

This module consists of the templates for input/output (I/O) helper classes.

Attributes:
    SUPPORTED_VISUALIZATION_FORMATS (tuple[str]): Image formats that are
        supported for visualization.
"""
import enum
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Collection, Dict, Union

__all__ = ["ImageDataFormat", "DataReader", "DataWriter", "SUPPORTED_VISUALIZATION_FORMATS"]

SUPPORTED_VISUALIZATION_FORMATS = (
    "png",
    "eps",
    "pdf",
    "jpeg",
    "pgf",
    "ps",
    "raw",
    "rgba",
    "svg",
    "svgz",
    "tiff",
)


class ImageDataFormat(enum.Enum):
    """Enum describing supported data formats for medical volume I/O."""

    nifti = 1, ("nii", "nii.gz")
    dicom = 2, ("dcm",)

    def __new__(cls, key_code, extensions):
        """
        Args:
            key_code (int): Enum value.
            extensions (tuple[str]): Extensions supported by format.
        """
        obj = object.__new__(cls)
        obj._value_ = key_code
        obj.extensions = extensions
        return obj

    def is_filetype(self, file_path: Union[str, Path, os.PathLike]) -> bool:
        """Verify if file path matches the file type specified by ImageDataFormat.

        This method checks to make sure the extensions are appropriate.

        Args:
            file_path (str): File path.

        Returns:
            bool: True if file_path has valid extension, False otherwise.
        """
        file_path = str(file_path)
        bool_list = [file_path.endswith(".%s" % ext) for ext in self.extensions]

        return bool(sum(bool_list))

    @classmethod
    def get_image_data_format(cls, file_or_dir_path: Union[str, Path, os.PathLike]):
        """Get the `ImageDataFormat` that corresponds to the file path.

        Matches extension to file path. If input is a directory path, then
        it is classified as `ImageDataFormat.dicom`.

        Args:
            file_or_dir_path (str): Path to a file or a directory.

        Returns:
            ImageDataFormat: Format corresponding to file or directory path.

        Raises:
            ValueError: If no compatible ImageDataFormat found.
        """
        for im_data_format in cls:
            if im_data_format.is_filetype(file_or_dir_path):
                return im_data_format

        # if no extension found, assume the name corresponds to a directory
        # and assume that format is dicom.
        # We cannot check if the path is a directory path because it may not
        # have been created yet.
        file_or_dir_path = str(file_or_dir_path)
        filename_base, ext = os.path.splitext(file_or_dir_path)
        if filename_base == file_or_dir_path:
            return ImageDataFormat.dicom

        raise ValueError(f"Unknown data format for {file_or_dir_path}")


class _StateMixin(ABC):
    """Temporary mixin that supports fetching and loading from state dictionaries.

    Note:
        This functionality is not well supported and will likely undergo changes
        in the alpha development process. Use with caution.
    """

    @abstractmethod
    def __serializable_variables__(self) -> Collection[str]:
        """Collection of serializable variables (i.e. state keys).

        This method should be implemented by subclasses and should
        return what parameters can be serialized. The values should
        typically be of a raw type.

        Returns:
            Collection[str]: Serializable variables.
        """
        raise NotImplementedError  # pragma: no cover

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary for an object.

        Keys for this dictionary are specified by the object's
        :func:`self.__serializable_variables__()` method.

        Returns:
            Dict[str, Any]: The state dictionary.
        """
        return {k: self.__dict__[k] for k in self.__serializable_variables__()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state dictionary into the object.

        Args:
            state_dict (Dict[str, Any]): The state dictionary created using
                :func:`self.state_dict()`.

        Raises:
            AttributeError: If any key in ``state_dict`` is not an attribute in
                ``self``.
        """
        for k, v in state_dict.items():
            if not hasattr(self, k):
                raise AttributeError(f"{type(self)} does not have attribute '{k}'")
            setattr(self, k, v)


class DataReader(_StateMixin):
    """Abstract class for reading medical data.

    Format-specific readers should inherit from this class.

    Attributes:
        data_format_code (ImageDataFormat): Should be defined by subclasses.
    """

    data_format_code = None

    @abstractmethod
    def load(self, file_path: str):
        """Load volume.

        Args:
            file_path (str): File path to load volume from.

        Returns:
            MedicalVolume: The loaded volume.
        """
        pass  # pragma: no cover

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`self.load`."""
        return self.load(*args, **kwargs)


class DataWriter(_StateMixin):
    """Abstract class for writing medical data.

    Format-specific writers should inherit from this class.

    Attributes:
        data_format_code (ImageDataFormat): Should be defined by subclasses.
    """

    data_format_code = None

    @abstractmethod
    def save(self, volume, file_path: str):
        """Save volume.

        Args:
            volume (MedicalVolume): Volume to save.
            file_path (str): File path to save volume to.
        """
        pass  # pragma: no cover

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`self.save`."""
        return self.save(*args, **kwargs)
