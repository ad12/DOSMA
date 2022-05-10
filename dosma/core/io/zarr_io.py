"""Zarr I/O.

This module contains Zarr input/output helpers.
"""

from typing import Collection

from dosma.core.io.format_io import DataReader, DataWriter, ImageDataFormat
from dosma.core.med_volume import MedicalVolume

__all__ = ["ZarrReader", "ZarrWriter"]


class ZarrReader(DataReader):
    """A class for reading Zarr files.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.zarr

    def load(self, **kwargs) -> MedicalVolume:
        """Load volume from Zarr store.

        A Zarr store should only correspond to one volume.

        Args:
            **kwargs: Parameters are passed through to `MedicalVolume.from_zarr`.

        Returns:
            MedicalVolume: Loaded volume.
        """

        return MedicalVolume.from_zarr(**kwargs)

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover


class ZarrWriter(DataWriter):
    """A class for writing volumes to Zarr stores.

    Attributes:
        data_format_code (ImageDataFormat): The supported image data format.
    """

    data_format_code = ImageDataFormat.zarr

    def save(self, volume: MedicalVolume, **kwargs):
        """Save volume in Zarr format.

        Args:
            **kwargs: Parameters are passed through to `MedicalVolume.from_zarr`.
        """
        volume.to_zarr(**kwargs)

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    write = save  # pragma: no cover
