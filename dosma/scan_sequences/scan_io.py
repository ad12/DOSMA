import inspect
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Union

import pydicom

from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.dicom_io import DicomReader
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils


def _contains_type(value, types):
    """Returns ``True`` if any value is an instance of ``types``."""
    if isinstance(value, types):
        return True
    if not isinstance(value, str) and isinstance(value, (Sequence, Set)) and len(value) > 0:
        return any(_contains_type(x, types) for x in value)
    elif isinstance(value, Dict):
        return _contains_type(value.keys(), types) or _contains_type(value.values(), types)
    return isinstance(value, types)


class ScanIOMixin(ABC):
    # This is just a summary on variables used in this abstract class,
    # the proper values/initialization should be done in child class.
    NAME: str
    __DEFAULT_SPLIT_BY__: Optional[str]
    _from_file_args: Dict[str, Any]

    @classmethod
    def from_dicom(
        cls,
        dir_or_files,
        group_by=None,
        ignore_ext: bool = False,
        num_workers: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        """Load scan from dicom files.

        Args:
            dir_or_files (str): The path to dicom directory or files.
            group_by: DICOM field tag name or tag number used to group dicoms. Defaults
                to scan's ``__DEFAULT_SPLIT_BY__``.
            ignore_ext (bool, optional): If `True`, ignore extension (`.dcm`)
                when loading dicoms from directory.
            num_workers (int, optional): Number of workers to use for loading.
            verbose (bool, optional): If ``True``, enable verbose logging for dicom loading.
            kwargs: Other keywords required to construct scan.

        Returns:
            The scan.
        """
        dr = DicomReader(num_workers, verbose)
        if group_by is None:
            group_by = cls.__DEFAULT_SPLIT_BY__
        volumes = dr.load(dir_or_files, group_by, ignore_ext)

        if isinstance(dir_or_files, (str, Path, os.PathLike)):
            dir_or_files = os.path.abspath(dir_or_files)
        else:
            dir_or_files = type(dir_or_files)([os.path.abspath(x) for x in dir_or_files])

        scan = cls(volumes, **kwargs)
        scan._from_file_args = {
            "dir_or_files": dir_or_files,
            "ignore_ext": ignore_ext,
            "group_by": group_by,
            "_type": "dicom",
        }

        return scan

    @classmethod
    def from_dict(cls, data: Dict[str, Any], force: bool = False):
        """Loads class from data dictionary.

        Args:
            data (Dict): The data.
            force (bool, optional): If ``True``, writes attributes even if they do not exist.
                Use with caution.

        Returns:
            The scan

        Examples:
            >>> scan = ... # some scan
            >>> filepath = scan.save("/path/to/base/directory")
            >>> scan_from_saved = type(scan).from_dict(io_utils.load_pik(filepath))
            >>> scan_from_dict = type(scan).from_dict(scan.__dict__)
        """
        # TODO: Add check for deprecated and converted attribute names.
        data = cls._convert_attr_name(data)

        # TODO: Convert metadata to appropriate type.
        # Converting metadata type is important when loading MedicalVolume data (for example).
        # The data is stored as a path, but should be loaded as a MedicalVolume.
        data = cls.load_custom_data(data)

        signature = inspect.signature(cls)
        init_metadata = {k: v for k, v in data.items() if k in signature.parameters}
        scan = cls(**init_metadata)
        for k in init_metadata.keys():
            data.pop(k)

        for k, v in data.items():
            if not hasattr(scan, k) and not force:
                warnings.warn(
                    f"{cls.__name__} does not have attribute {k}. Skipping...", stacklevel=2
                )
                continue
            scan.__setattr__(k, v)

        return scan

    def save(
        self,
        path: str,
        save_custom: bool = False,
        image_data_format: ImageDataFormat = None,
        num_workers: int = 0,
    ):
        """Saves scan data to disk with option for custom saving.

        Custom saving may be useful to reduce redundant saving and/or save data in standard
        compatible formats (e.g. medical images - nifti/dicom), which are not feasible with
        python serialization libraries, like pickle.

        When ``save_custom=True``, this method overloads standard pickling with customizable
        saving by first saving data in customizable way (e.g. MedicalVolume -> Nifti file),
        and then pickling the reference to the saved object (e.g. Nifti filepath).

        Currently certain custom saving of objects such as ``pydicom.FileDataset`` and
        :cls:`Tissue` objects are not supported.

        To load the data, do the following:

        >>> filepath = scan.save("/path/to/directory", save_custom=True)
        >>> scan_loaded = type(scan).load(io_utils.load_pik(filepath))

        Args:
            path (str): Directory where data is stored.
            data_format (ImageDataFormat, optional): Format to save data.
                Defaults to ``preferences.image_data_format``.
            save_custom (bool, optional): If ``True``, saves data in custom way specified
                by :meth:`save_custom_data` in format specified by ``data_format``. For
                example, for default classes this will save :cls:`MedicalVolume` data
                to nifti/dicom files as specified by ``image_data_format``.
            image_data_format (ImageDataFormat, optional): The data format to save
                :cls:`MedicalVolume` data. Only used if save_custom is ``True``.
            num_workers (int, bool): Number of workers for saving custom data.
                Only used if save_custom is ``True``.

        Returns:
            str: The path to the pickled file.
        """
        if image_data_format is None:
            image_data_format = preferences.image_data_format

        save_dirpath = path  # self._save_dir(path)
        os.makedirs(save_dirpath, exist_ok=True)
        filepath = os.path.join(save_dirpath, "%s.data" % self.NAME)

        metadata: Dict = {}
        for attr in self.__serializable_variables__():
            metadata[attr] = self.__getattribute__(attr)

        if save_custom:
            metadata = self._save(
                metadata, save_dirpath, image_data_format=image_data_format, num_workers=num_workers
            )

        io_utils.save_pik(filepath, metadata)
        return filepath

    @classmethod
    def load(cls, path_or_data: Union[str, Dict], num_workers: int = 0):
        """Load scan.

        This method overloads the :func:`from_dict` method by supporting loading from a file
        in addition to the data dictionary. If loading and constructing a scan using
        :func:`from_dict` fails, defaults to loading data from original dicoms
        (if ``self._from_file_args`` is initialized).

        Args:
            path_or_data (Union[str, Dict]): Pickle file to load or data dictionary.
            num_workers (int, optional): Number of workers to use for loading.

        Returns:
            ScanSequence: Of type ``cls``.

        Raises:
            ValueError: If ``scan`` cannot be constructed.
        """
        if isinstance(path_or_data, (str, Path, os.PathLike)):
            if os.path.isdir(path_or_data):
                path_or_data = os.path.join(path_or_data, f"{cls.NAME}.data")

            if not os.path.isfile(path_or_data):
                raise FileNotFoundError(f"File {path_or_data} does not exist")
            data = io_utils.load_pik(path_or_data)
        else:
            data = path_or_data

        try:
            scan = cls.from_dict(data)
            return scan
        except Exception:
            warnings.warn(
                f"Failed to load {cls.__name__} from data. Trying to load from dicom file.",
                stacklevel=2,
            )

        data = cls._convert_attr_name(data)
        data = cls.load_custom_data(data, num_workers=num_workers)

        scan = None
        if "_from_file_args" in data:
            dicom_args = data.pop("_from_file_args")
            assert dicom_args.pop("_type") == "dicom"
            scan = cls.from_dicom(**dicom_args, num_workers=num_workers)
        elif "dicom_path" in data:
            # Backwards compatibility
            dicom_path = data.pop("dicom_path")
            ignore_ext = data.pop("ignore_ext", False)
            group_by = data.pop("split_by", cls.__DEFAULT_SPLIT_BY__)
            scan = cls.from_dicom(
                dicom_path, ignore_ext=ignore_ext, group_by=group_by, num_workers=num_workers
            )

        if scan is None:
            raise ValueError(f"Data is insufficient to construct {cls.__name__}")

        for k, v in data.items():
            if not hasattr(scan, k):
                warnings.warn(
                    f"{cls.__name__} does not have attribute {k}. Skipping...", stacklevel=2
                )
                continue
            scan.__setattr__(k, v)

        return scan

    def save_data(
        self, base_save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format
    ):
        """Deprecated: Alias for :func:`self.save`."""
        warnings.warn(
            "save_data is deprecated since v0.0.13 and will no longer be "
            "available in v0.1. Use `save` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.save(base_save_dirpath, data_format)

    def _save(
        self,
        metadata: Dict[str, Any],
        save_dir: str,
        fname_fmt: Dict[Union[str, type], str] = None,
        **kwargs,
    ):
        if fname_fmt is None:
            fname_fmt = {}

        default_fname_fmt = {MedicalVolume: "image-{}"}
        for k, v in default_fname_fmt.items():
            if k not in fname_fmt:
                fname_fmt[k] = v

        for attr in metadata.keys():
            val = metadata[attr]
            path = fname_fmt.get(attr, None)

            if path is None:
                path = os.path.abspath(os.path.join(save_dir, attr))
            if not os.path.isabs(path):
                path = os.path.join(save_dir, attr, path)
            try:
                metadata[attr] = self.save_custom_data(val, path, fname_fmt, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to save metadata {attr} - {e}")

        return metadata

    def save_custom_data(
        self, metadata, paths, fname_fmt: Dict[Union[str, type], str] = None, **kwargs
    ):
        """
        Finds all attributes of type MedicalVolume or Sequence/Mapping to MedicalVolume
        and saves them.
        """
        if isinstance(metadata, (Dict, Sequence, Set)):
            if isinstance(paths, str):
                paths = [paths] * len(metadata)
            else:
                assert len(paths) == len(metadata)

        if isinstance(metadata, Dict):
            keys = metadata.keys()
            if isinstance(paths, Dict):
                paths = [paths[k] for k in keys]
            paths = [os.path.join(_path, f"{k}") for k, _path in zip(keys, paths)]
            values = self.save_custom_data(metadata.values(), paths, fname_fmt, **kwargs)
            metadata = dict(zip(keys, values))
        elif not isinstance(metadata, str) and isinstance(metadata, (Sequence, Set)):
            values = list(metadata)
            paths = [os.path.join(_path, "{:03d}".format(i)) for i, _path in enumerate(paths)]
            values = [
                self.save_custom_data(_x, _path, fname_fmt, **kwargs)
                for _x, _path in zip(values, paths)
            ]
            if not isinstance(values, type(metadata)):
                metadata = type(metadata)(values)
            else:
                metadata = values
        else:
            formatter = [fname_fmt.get(x) for x in type(metadata).__mro__]
            formatter = [x for x in formatter if x is not None]
            if len(formatter) == 0:
                formatter = None
            else:
                formatter = formatter[0]
            metadata = self._save_custom_data_base(metadata, paths, formatter, **kwargs)

        return metadata

    def _save_custom_data_base(self, metadata, path, formatter: str = None, **kwargs):
        """The base condition for :meth:`save_custom_data`.

        Args:
            metadata (Any): The data to save.
            path (str): The path to save the data.
            formatter (str, optional): If provided, this formatted string
                will be used to format ``path``.
        """
        out = {"__dtype__": type(metadata)}
        # TODO: Add support for num workers.
        # num_workers = kwargs.pop("num_workers", 0)

        if formatter:
            path = os.path.join(os.path.dirname(path), formatter.format(os.path.basename(path)))

        if isinstance(metadata, MedicalVolume):
            image_data_format = kwargs.get("image_data_format", preferences.image_data_format)
            # TODO: Once, `files` property added to MedicalVolume, check if property is
            # set before doing saving.
            path = fio_utils.convert_image_data_format(path, image_data_format)
            metadata.save_volume(path, data_format=image_data_format)
            out["__value__"] = path
        else:
            out = metadata

        return metadata

    @classmethod
    def _convert_attr_name(cls, data: Dict[str, Any]):
        return data

    @classmethod
    def load_custom_data(cls, data: Any, **kwargs):
        """Recursively converts data to appropriate types.

        By default, this loads all :class:`MedicalVolume` objects from their corresponding paths.

        Args:
            data (Any): The data. Can either be the dictionary or the metadata value.
                If the data corresponds to a custom type, it should have the following
                schema:

                {
                    '__dtype__': The type of the data.
                    '__value__': The value from which object of type __dtype__ can be constructed.
                }

            **kwargs: Keyword Arguments to pass to :meth:`_load_custom_data_base`.

        Returns:
            Any: The loaded metadata.
        """
        dtype = type(data)
        if isinstance(data, Dict) and "__value__" in data:
            dtype = data["__dtype__"]
            data = data["__value__"]

        if issubclass(dtype, Dict):
            keys = cls.load_custom_data(data.keys(), **kwargs)
            values = cls.load_custom_data(data.values(), **kwargs)
            data = dict(zip(keys, values))
        elif not issubclass(dtype, str) and issubclass(dtype, (list, tuple, set)):
            data = dtype([cls.load_custom_data(x, **kwargs) for x in data])
        else:
            data = cls._load_custom_data_base(data, dtype, **kwargs)

        return data

    @classmethod
    def _load_custom_data_base(cls, data, dtype=None, **kwargs):
        """The base condition for :meth:`load_custom_data`.

        Args:
            data:
            dtype (type): The data type.

        Return:
            The loaded data.
        """
        if dtype is None:
            dtype = type(data)

        # TODO: Add support for loading with num_workers
        num_workers = kwargs.get("num_workers", 0)
        if isinstance(data, str) and issubclass(dtype, MedicalVolume):
            data = fio_utils.generic_load(data, num_workers=num_workers)

        return data

    def __serializable_variables__(
        self, ignore_types=(pydicom.FileDataset, pydicom.Dataset, Tissue), ignore_attrs=()
    ) -> Set:
        """
        By default, all instance attributes are serialized except those
        corresponding to headers, :class:`MedicalVolume`(s), or :class:`Tissues`.
        Properties and class attributes are also not stored. Class attributes are
        indentified using the PEP8 nomenclature of all caps variables.

        Note:
            This method has not been profiled, but times may be large if
            the instance contains many variables. Currently this is not
            cached as attributes values can change and, as a result, must
            be inspected.
        """
        serializable = []
        for attr, value in self.__dict__.items():
            if attr in ignore_attrs or _contains_type(value, ignore_types):
                continue
            if attr.startswith("temp") or attr.startswith("_temp"):
                continue
            if attr.upper() == attr or (attr.startswith("__") and attr.endswith("__")):
                continue
            if callable(value) or isinstance(value, property):
                continue
            serializable.append(attr)

        return set(serializable)
