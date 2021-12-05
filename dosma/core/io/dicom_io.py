"""
DICOM I/O.

This module contains DICOM input/output helpers.

Note:

    1. Dicom utilizes LPS convention:
        - LPS: right --> left, anterior --> posterior, inferior --> superior
        - we will call it LPS+, such that letters correspond to increasing end of axis

Attributes:
    TOTAL_NUM_ECHOS_KEY (tuple[int]): Hexadecimal encoding of DICOM tag corresponding
        to number of echos.
"""

import copy
import functools
import itertools
import multiprocessing as mp
import os
import re
from math import ceil, log10
from typing import Collection, List, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pydicom
from natsort import index_natsorted, natsorted
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from dosma.core import orientation as stdo
from dosma.core.io.format_io import DataReader, DataWriter, ImageDataFormat
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import AFFINE_DECIMAL_PRECISION, SCANNER_ORIGIN_DECIMAL_PRECISION

__all__ = ["DicomReader", "DicomWriter"]

TOTAL_NUM_ECHOS_KEY = (0x19, 0x107E)
PATH_LIKE = (str, os.PathLike)


class DicomReader(DataReader):
    """A class for reading DICOM files.

    Attributes:
        num_workers (int, optional): Number of workers to use for loading.
        verbose (bool, optional): If ``True``, show loading progress bar.
        group_by (str(s) or int(s), optional): DICOM attribute(s) used
            to group dicoms. This can be the attribute tag name (str) or tag
            number (int).
        sort_by (str(s) or int(s), optional): DICOM attribute(s) used
            to sort dicoms. This sorting is done after sorting files in alphabetical
            order.
        ignore_ext (bool, optional): If ``True``, ignore extension (``".dcm"``)
            when loading dicoms from directory.
        default_ornt (Tuple[str, str], optional): Default in-plane orientation to use if
            orientation cannot be determined from DICOM header. If not specified
            and orientation cannot be determined, error will be raised.
        data_format_code (ImageDataFormat): The supported image data format.

    Examples:
        >>> # Load single dicom
        >>> dr = DicomReader()
        >>> mv = dr.load("/path/to/dicom/file", group_by=None)[0]

        >>> # Load multi-echo MRI data
        >>> dr = DicomReader(num_workers=0, verbose=True)
        >>> mvs = dr.load("/dicoms/directory", group_by="EchoTime", sort_by="InstanceNumber")

        >>> # Use the same loader for multiple multi-echo time-series MRI scans
        >>> dr = DicomReader(group_by=["EchoTime", "TriggerTime", sort_by="InstanceNumber")
        >>> scans = [dr.load(dcm_dir) for dcm_dir in ["/dicom/dir1", "/dicom/dir2", "/dicom/dir3"]]
    """

    data_format_code = ImageDataFormat.dicom

    def __init__(
        self,
        num_workers: int = 0,
        verbose: bool = False,
        group_by: Union[str, int, Sequence[Union[str, int]]] = "EchoNumbers",
        sort_by: Union[str, int, Sequence[Union[str, int]]] = None,
        ignore_ext: bool = False,
        default_ornt: Tuple[str, str] = None,
    ):
        """
        Args:
            num_workers (int, optional): Number of workers to use for loading.
            verbose (bool, optional): If ``True``, show loading progress bar.
            group_by (:obj:`str(s)` or :obj:`int(s)`, optional): DICOM attribute(s) used
                to group dicoms. This can be the attribute tag name (str) or tag
                number (int).
            sort_by (:obj:`str(s)` or :obj:`int(s)`, optional): DICOM attribute(s) used
                to sort dicoms. This sorting is done after sorting files in alphabetical
                order.
            ignore_ext (bool, optional): If ``True``, ignore extension (``".dcm"``)
                when loading dicoms from directory.
            default_ornt (Tuple[str, str], optional): Default in-plane orientation to use if
                orientation cannot be determined from DICOM header. If not specified
                and orientation cannot be determined, error will be raised.
        """
        self.num_workers = num_workers
        self.verbose = verbose
        self.group_by = group_by
        self.sort_by = sort_by
        self.ignore_ext = ignore_ext
        self.default_ornt = default_ornt

    def get_files(
        self,
        path,
        include: Union[str, Sequence[str]] = None,
        exclude: Union[str, Sequence[str]] = None,
        ignore_hidden: bool = True,
        ignore_ext: bool = np._NoValue,
    ):
        """Get dicom files from directory.

        Args:
            path (str): Directory with dicom file(s).
            include (str(s), optional): Regex pattern(s) of files to include.
                Pattern matching will only be applied to the base file name.
                Used with :func:`re.match`.
            exclude (str(s), optional): Regex pattern(s) of files to exclude.
                Pattern matching will only be applied to the base file name.
                Used with :func:`re.match`.
            ignore_ext (bool, optional): If ``True``, ignore extension (`.dcm`)
                when loading dicoms from directory.
            ignore_hidden (bool, optional): If ``True``, ignores hidden
                files (files starting with ``"."``). Defaults to ``self.ignore_ext``.

        Returns:
            List[str]: Dicom file paths (in natsort order)

        Raises:
            NotADirectoryError: If ``path`` does not correspond to directory.
        """
        if not os.path.isdir(path):
            raise NotADirectoryError("`path` must be path to directory with dicoms.")

        ignore_ext = ignore_ext if ignore_ext != np._NoValue else self.ignore_ext

        include = _wrap_as_tuple(include, default=())
        exclude = _wrap_as_tuple(exclude, default=())
        assert isinstance(include, tuple)
        assert isinstance(exclude, tuple)
        if ignore_hidden:
            exclude += ("^\.",)  # noqa: W605

        possible_files = os.listdir(path)
        lstFilesDCM = []
        for f in possible_files:
            # If ignore extension, don't look for '.dcm' extension.
            is_file = os.path.isfile(os.path.join(path, f))
            match_ext = ignore_ext or self.data_format_code.is_filetype(f)
            if (
                is_file
                and match_ext
                and (not include or any(re.match(x, f) for x in include))
                and (not exclude or all(not re.match(x, f) for x in exclude))
            ):
                lstFilesDCM.append(os.path.join(path, f))

        lstFilesDCM = natsorted(lstFilesDCM)
        return lstFilesDCM

    def _handle_files(self, path, ignore_ext):
        """Gets and organize pydicom data from file(s) or directory.

        Args:
            path (str | path-like): The file path(s) or directory path to
                load from.
            ignore_ext (bool): If ``True``, ignore extension (``".dcm"``)
                when loading dicoms from directory.

        Returns:
            Sequence[path-like]: The filepaths for different slices.
        """
        if isinstance(path, str) or not isinstance(path, Sequence):
            if os.path.isdir(path):
                lstFilesDCM = self.get_files(path, ignore_hidden=True, ignore_ext=ignore_ext)
            elif os.path.isfile(path):
                lstFilesDCM = [path]
            else:
                raise IOError(f"No directory or file found - {path}")
        else:
            not_files = [x for x in path if not os.path.isfile(x)]
            if len(not_files) > 0:
                raise IOError(
                    "Files not found:\n{}".format("".join("\t{}\n".format(x) for x in not_files))
                )
            lstFilesDCM = path

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No valid dicom files found in {}".format(path))

        return lstFilesDCM

    def load(
        self,
        path_or_bytes: Union[str, os.PathLike, bytes, Sequence[Union[str, os.PathLike, bytes]]],
        group_by: Union[str, int, Sequence[Union[str, int]]] = np._NoValue,
        sort_by: Union[str, int, Sequence[Union[str, int]]] = np._NoValue,
        ignore_ext: bool = np._NoValue,
        default_ornt: Tuple[str, str] = np._NoValue,
    ):
        """Load dicoms into ``MedicalVolume``s grouped by ``group_by`` tag(s).

        When loading files from a directory, all hidden files (files starting with ``"."``)
        are ignored. Files are initially sorted in alphabetical order and subsequently by
        ``sort_by`` if specified.

        Args:
            path_or_bytes (`str(s)`): Directory with dicom files or dicom file(s).
                Dicom file(s) can either be the path or the bytes from the opened file.
            group_by (:obj:`str(s)` or :obj:`int(s)`, optional): DICOM attribute(s) used
                to group dicoms. This can be the attribute tag name (str) or tag
                number (int). Defaults to ``self.group_by``.
            sort_by (:obj:`str(s)` or :obj:`int(s)`, optional): DICOM attribute(s) used
                to sort dicoms. This sorting is done after sorting files in alphabetical
                order. Defaults to ``self.sort_by``.
            ignore_ext (bool, optional): If ``True``, ignore extension (``".dcm"``)
                when loading dicoms from directory. Defaults to ``self.ignore_ext``.
            default_ornt (Tuple[str, str], optional): Default in-plane orientation to use if
                orientation cannot be determined from DICOM header. If not specified
                and orientation cannot be determined, error will be raised.
                Defaults to ``self.default_ornt``.

        Returns:
            list[MedicalVolume]: Different volumes grouped by the `group_by` DICOM tag.

        Raises:
            ValueError: If `group_by` not specified or if single dicom file specified.
            IOError: If directory or dicom file(s) specified by `path` do not exist.
            FileNotFoundError: If no valid dicom files found.

        Note:
            Not setting the ``group_by`` argument could result in ill-formatted volumes.
            For best performance, specify ``group_by`` based on the attribute(s) differentiating
            different volumes in the scan.
        """
        group_by = group_by if group_by != np._NoValue else self.group_by
        sort_by = sort_by if sort_by != np._NoValue else self.sort_by
        ignore_ext = ignore_ext if ignore_ext != np._NoValue else self.ignore_ext
        default_ornt = default_ornt if default_ornt != np._NoValue else self.default_ornt

        group_by = _wrap_as_tuple(group_by, default=())
        sort_by = _wrap_as_tuple(sort_by, default=())

        if isinstance(path_or_bytes, PATH_LIKE) or (
            isinstance(path_or_bytes, Sequence) and isinstance(path_or_bytes[0], PATH_LIKE)
        ):
            lstFilesDCM = self._handle_files(path_or_bytes, ignore_ext)
        else:
            # We explicitly specify list/tuple because other objects can be sequences
            # (e.g. bytes) that we cannot thoroughly account for.
            lstFilesDCM = (
                [path_or_bytes] if not isinstance(path_or_bytes, (list, tuple)) else path_or_bytes
            )

        if self.num_workers:
            fn = functools.partial(pydicom.read_file, force=True)
            if self.verbose:
                dicom_slices = process_map(fn, lstFilesDCM, max_workers=self.num_workers)
            else:
                with mp.Pool(self.num_workers) as p:
                    dicom_slices = p.map(fn, lstFilesDCM)
        else:
            dicom_slices = [
                pydicom.read_file(fp, force=True)
                for fp in tqdm(lstFilesDCM, disable=not self.verbose)
            ]

        # Check if dicom file has the group_by element specified
        temp_dicom = dicom_slices[0]
        for _group in group_by:
            if _group not in temp_dicom:
                raise KeyError("Tag {} does not exist in dicom".format(_group))

        if sort_by:
            try:
                dicom_slices = natsorted(
                    dicom_slices,
                    key=lambda x: tuple(
                        _unpack_dicom_attr(x, attr, required=True) for attr in sort_by
                    ),
                )
            except KeyError as e:
                raise KeyError(f"Tag not found in dicom - {e}")

        dicom_data = {}
        for ds in dicom_slices:
            val_groupby = tuple(_unpack_dicom_attr(ds, attr, required=True) for attr in group_by)
            if val_groupby not in dicom_data.keys():
                dicom_data[val_groupby] = {"headers": [], "arr": []}

            dicom_data[val_groupby]["headers"].append(ds)
            dicom_data[val_groupby]["arr"].append(ds.pixel_array)

        vols = []
        for k in sorted(dicom_data.keys()):
            dd = dicom_data[k]
            headers = dd["headers"]
            if len(headers) == 0:
                continue
            arr = np.stack(dd["arr"], axis=-1)

            affine = to_RAS_affine(headers, default_ornt=default_ornt)

            vol = MedicalVolume(arr, affine, headers=headers)
            vols.append(vol)

        return vols

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    read = load  # pragma: no cover


class DicomWriter(DataWriter):
    """A class for writing volumes in DICOM format.

    Attributes:
        num_workers (int, optional): Number of workers to use for writing.
        verbose (bool, optional): If ``True``, show writing progress bar.
        fname_fmt (str, optional): Formatting string for filenames.
        sort_by (:obj:`str(s)` or :obj:`int(s)`, optional): DICOM attribute(s) used
            to define ordering of slices prior to writing. If not specified, this ordering
            will be defined by the order of blocks in ``volume``.
        data_format_code (ImageDataFormat): The supported image data format.

    Examples:
        >>> # Save MedicalVolume mv
        >>> dw = DicomWriter()
        >>> dw.save(mv, "/path/to/save/folder")

        >>> dw = DicomWriter(fname_fmt="I%05d.dcm", sort_by="InstanceNumber")
        >>> dw.save(mv, "/path/to/save/folder")
    """

    data_format_code = ImageDataFormat.dicom

    def __init__(
        self,
        num_workers: int = 0,
        verbose: bool = False,
        fname_fmt: str = None,
        sort_by: Union[str, int, Sequence[Union[str, int]]] = None,
    ):
        """

        Args:
            num_workers (int, optional): Number of workers to use for writing.
            verbose (bool, optional): If ``True``, show writing progress bar.
            fname_fmt (str, optional): Formatting string for filenames. Must contain ``%d``,
                which correspopnds to slice number. Defaults to
                ``"I%0{max(4, ceil(log10(num_slices)))}d.dcm"`` (e.g. ``"I0001.dcm"``).
            sort_by (str(s) or int(s), optional): DICOM attribute(s) used
                to define ordering of slices prior to writing. If not specified, this ordering
                will be defined by the order of blocks in ``volume``.
        """
        self.num_workers = num_workers
        self.verbose = verbose
        self.fname_fmt = fname_fmt
        self.sort_by = sort_by

    def save(
        self,
        volume: MedicalVolume,
        dir_path: str,
        fname_fmt: str = np._NoValue,
        sort_by: Union[str, int, Sequence[Union[str, int]]] = np._NoValue,
    ):
        """Save `medical volume` in dicom format.

        This function assumes headers for the volume (``volume.headers()``) exist
        for one spatial dimension. Headers for non-spatial dimensions are optional, but
        highly recommended. If provided, they will be used to write the volume. If not,
        headers will be appropriately broadcast to these dimensions. Note, this means
        that multiple files will have the same header information and will not be able
        to be loaded automatically.

        Currently header spatial information (orientation, origin, slicing between spaces,
        etc.) is not overwritten nor validated. All data must correspond to the same
        spatial information as specified in the headers to produce valid DICOM files.

        Args:
            volume (MedicalVolume): Volume to save.
            dir_path: Directory path to store dicom files. Dicoms are stored in directories,
                as multiple files are needed to store the volume.
            fname_fmt (str, optional): Formatting string for filenames. Must contain ``%d``,
                which correspopnds to slice number. Defaults to ``self.fname_fmt``.
            sort_by (``str``(s) or ``int``(s), optional): DICOM attribute(s) used
                to define ordering of slices prior to writing. If ``None``, this ordering
                will be defined by the order of blocks in ``volume``. Defaults to
                ``self.sort_by``.

        Raises:
            ValueError: If `im` does not have initialized headers. Or if `im` was flipped across
                any axis. Flipping changes scanner origin, which is currently not handled.
        """
        fname_fmt = fname_fmt if fname_fmt != np._NoValue else self.fname_fmt
        sort_by = sort_by if sort_by != np._NoValue else self.sort_by

        # Get orientation indicated by headers.
        headers = volume.headers()
        if headers is None:
            raise ValueError("MedicalVolume headers must be initialized to save as a dicom")

        sort_by = _wrap_as_tuple(sort_by, default=())

        # Reformat to put headers in last dimensions.
        single_dim = []
        full_dim = []
        for i, dim in enumerate(headers.shape[:3]):
            if dim == 1:
                single_dim.append(i)
            else:
                full_dim.append(i)
        if len(full_dim) > 1:
            raise ValueError(
                f"Only one spatial dimension can have headers. Got {len(full_dim)} - "
                f"headers.shape={headers.shape[:3]}"
            )
        new_orientation = (volume.orientation[x] for x in single_dim + full_dim)

        volume = volume.reformat(new_orientation)
        assert volume.headers().shape[:3] == (1, 1, volume.shape[2])

        # Reformat medical volume to expected orientation specified by dicom headers.
        # NOTE: This is temporary. Future fixes will allow us to modify header
        # data to match affine matrix.
        if len(volume.shape) > 3:
            shape = volume.shape[3:]
            multi_volumes = np.empty(shape, dtype=object)
            for dims in itertools.product(*[list(range(0, x)) for x in multi_volumes.shape]):
                multi_volumes[dims] = _format_volume_to_header(volume[(Ellipsis,) + dims])
            multi_volumes = multi_volumes.flatten()
            volume_arr = np.concatenate([v.volume for v in multi_volumes], axis=-1)
            headers = np.concatenate([v.headers(flatten=True) for v in multi_volumes], axis=-1)
        else:
            volume = _format_volume_to_header(volume)
            volume_arr = volume.volume
            headers = volume.headers(flatten=True)

        assert headers.ndim == 1
        assert volume_arr.shape[2] == len(
            headers
        ), "Dimension mismatch - {:d} slices but {:d} headers".format(
            volume_arr.shape[-1], len(headers)
        )

        if sort_by:
            idxs = np.asarray(
                index_natsorted(
                    headers,
                    key=lambda h: tuple(_unpack_dicom_attr(h, k, required=True) for k in sort_by),
                )
            )
            headers = headers[idxs]
            volume_arr = volume_arr[..., idxs]

        # Check if dir_path exists.
        os.makedirs(dir_path, exist_ok=True)

        num_slices = len(headers)
        if not fname_fmt:
            filename_format = "I%0" + str(max(4, ceil(log10(num_slices)))) + "d.dcm"
        else:
            filename_format = fname_fmt

        filepaths = [os.path.join(dir_path, filename_format % (s + 1)) for s in range(num_slices)]
        if self.num_workers:
            slices = [volume_arr[..., s] for s in range(num_slices)]
            if self.verbose:
                process_map(_write_dicom_file, slices, headers, filepaths)
            else:
                with mp.Pool(self.num_workers) as p:
                    out = p.starmap_async(_write_dicom_file, zip(slices, headers, filepaths))
                    out.wait()
        else:
            for s in tqdm(range(num_slices), disable=not self.verbose):
                _write_dicom_file(volume_arr[..., s], headers[s], filepaths[s])

    def __serializable_variables__(self) -> Collection[str]:
        return self.__dict__.keys()

    write = save  # pragma: no cover


def to_RAS_affine(headers: List[pydicom.FileDataset], default_ornt: Tuple[str, str] = None):
    """Convert from LPS+ orientation (default for DICOM) to RAS+ standardized orientation.

    Args:
        headers (list[pydicom.FileDataset]): Headers for DICOM files to reorient.
            Files should correspond to single volume.

    Returns:
        np.ndarray: Affine matrix.
    """
    try:
        im_dir = headers[0].ImageOrientationPatient
    except AttributeError:
        im_dir = _decode_inplane_direction(headers, default_ornt=default_ornt)
        if im_dir is None:
            raise RuntimeError("Could not determine in-plane directions from headers.")
    try:
        in_plane_pixel_spacing = headers[0].PixelSpacing
    except AttributeError:
        try:
            in_plane_pixel_spacing = headers[0].ImagerPixelSpacing
        except AttributeError:
            raise RuntimeError(
                "Could not determine in-plane pixel spacing from headers. "
                "Neither attribute 'PixelSpacing' nor 'ImagerPixelSpacing' found."
            )

    orientation = np.zeros([3, 3])

    # Determine vector for in-plane pixel directions (i, j).
    i_vec, j_vec = (
        np.asarray(im_dir[:3]).astype(np.float64),
        np.asarray(im_dir[3:]).astype(np.float64),
    )  # unique to pydicom, please revise if using different library to load dicoms
    i_vec, j_vec = (
        np.round(i_vec, AFFINE_DECIMAL_PRECISION),
        np.round(j_vec, AFFINE_DECIMAL_PRECISION),
    )
    i_vec = i_vec * in_plane_pixel_spacing[0]
    j_vec = j_vec * in_plane_pixel_spacing[1]

    # Determine vector for through-plane pixel direction (k).
    # Compute difference in patient position between consecutive headers.
    # This is the preferred method to determine the k vector.
    # If single header, take cross product between i/j vectors.
    # These actions are done to avoid rounding errors that might result from float subtraction.
    if len(headers) > 1:
        k_vec = np.asarray(headers[1].ImagePositionPatient).astype(np.float64) - np.asarray(
            headers[0].ImagePositionPatient
        ).astype(np.float64)
    else:
        slice_thickness = headers[0].get("SliceThickness", 1.0)
        i_norm = 1 / np.linalg.norm(i_vec) * i_vec
        j_norm = 1 / np.linalg.norm(j_vec) * j_vec
        k_norm = np.cross(i_norm, j_norm)
        k_vec = k_norm / np.linalg.norm(k_norm) * slice_thickness
        if hasattr(headers[0], "SpacingBetweenSlices") and headers[0].SpacingBetweenSlices < 0:
            k_vec *= -1
    k_vec = np.round(k_vec, AFFINE_DECIMAL_PRECISION)

    orientation[:3, :3] = np.stack([j_vec, i_vec, k_vec], axis=1)
    scanner_origin = headers[0].get("ImagePositionPatient", np.zeros((3,)))
    scanner_origin = np.asarray(scanner_origin).astype(np.float64)
    scanner_origin = np.round(scanner_origin, SCANNER_ORIGIN_DECIMAL_PRECISION)

    affine = np.zeros([4, 4])
    affine[:3, :3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2, :] = -1 * affine[:2, :]
    affine[3, 3] = 1

    affine[affine == 0] = 0

    return affine


def _decode_inplane_direction(headers: Sequence[pydicom.FileDataset], default_ornt=None):
    """Helper function to decode in-plane direction from header(s).

    Recall the direction in dicoms are in cartesian order ``(x,y)``,
    but numpy/dosma are in matrix order ``(y,x)``. When adding new
    methods, make sure to account for this.

    Returns:
        np.ndarray: 6-element LPS direction array where first 3 elements define
            direction for x-direction (columns) and second 3 elements define
            direction for y-direction (rows)
    """
    _patient_ornt_to_nib = {"H": "S", "F": "I"}

    if (
        len(headers) == 1
        and hasattr(headers[0], "PatientOrientation")
        and headers[0].PatientOrientation
    ):
        # Decoder: patient orientation.
        # Patient orientation is only decoded along principal direction (e.g. "FR" -> "F").
        ornt = [
            _patient_ornt_to_nib.get(k[:1], k[:1]) for k in headers[0].PatientOrientation
        ]  # (x,y)
        ornt = stdo.orientation_nib_to_standard(ornt)
        affine = stdo.to_affine(ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    if default_ornt:
        affine = stdo.to_affine(default_ornt)
        affine[:2, :] = -1 * affine[:2, :]
        return np.concatenate([affine[:3, 0], affine[:3, 1]], axis=0)

    return None


def _format_volume_to_header(volume: MedicalVolume) -> MedicalVolume:
    """Reformats the volume according to its header.

    Args:
        volume (MedicalVolume): The volume to reformat.
            Must be 3D and have headers of shape (1, 1, volume.shape[2]).

    Returns:
        MedicalVolume: The reformatted volume.
    """
    headers = volume.headers()
    assert headers.shape == (1, 1, volume.shape[2])

    affine = to_RAS_affine(headers.flatten())
    orientation = stdo.orientation_nib_to_standard(nib.aff2axcodes(affine))

    # Currently do not support mismatch in scanner_origin.
    if tuple(affine[:3, 3]) != volume.scanner_origin:
        raise ValueError(
            "Scanner origin mismatch. "
            "Currently we do not handle mismatch in scanner origin "
            "(i.e. cannot flip across axis)"
        )

    volume = volume.reformat(orientation)
    assert volume.headers().shape == (1, 1, volume.shape[2])
    return volume


def _write_dicom_file(np_slice: np.ndarray, header: pydicom.FileDataset, file_path: str):
    """Replace data in header with 2D numpy array and write to `file_path`.

    Args:
        np_slice (np.ndarray): 2D slice to encode in dicom file.
        header (pydicom.FileDataset): DICOM header.
        file_path: File path to write to.
    """
    # Deep copy required in case headers are shared.
    header = copy.deepcopy(header)
    expected_dimensions = header.Rows, header.Columns
    assert (
        np_slice.shape == expected_dimensions
    ), "In-plane dimension mismatch - expected shape {}, got {}".format(
        str(expected_dimensions), str(np_slice.shape)
    )

    np_slice_bytes = np_slice.tobytes()
    bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)
    if bit_depth != header.BitsAllocated:
        np_slice = _update_np_dtype(np_slice, header.BitsAllocated)
        np_slice_bytes = np_slice.tobytes()
        bit_depth = int(len(np_slice_bytes) / (np_slice.shape[0] * np_slice.shape[1]) * 8)

    assert bit_depth == header.BitsAllocated, "Bit depth mismatch: Expected {:d} got {:d}".format(
        header.BitsAllocated, bit_depth
    )

    header.PixelData = np_slice_bytes

    header.save_as(file_path)


def _update_np_dtype(arr: np.ndarray, bit_depth: int):
    """Create copy of np_array with bit-depth and type specified here.

    Note pydicom only supports writing dicoms with bit-depth 8/16 - only supports bit depths 8/16.

    Args:
        arr (np.ndarray): Numpy array to put into given bit depth.
        bit_depth (int): Bit depth for writing dicom. Must be either `8` or `16`.

    Returns:
        np.ndarray: Copy of input np_array.

    Raises:
        ValueError: If `arr` out of bit-depth range.
        TypeError: If `arr` contains float values out of supported float types.
    """
    assert bit_depth in [8, 16], "Only bit-depths of 8 and 16 are currently supported."
    dtype_dict = {
        8: [(np.int8, -128, 127), (np.uint8, 0, 255)],
        16: [
            (np.uint16, 0, 2 ** 16 - 1),
            (np.int16, -(2 ** 15), 2 ** 15),
            (np.float16, -6.55e4, 6.55e4 - 1),
        ],
    }
    supported_floats = [np.float16]
    curr_min = np.min(arr)
    curr_max = np.max(arr)
    contains_float = (arr % 1 != 0).any()

    dtypes = dtype_dict[bit_depth]

    new_dtype = None
    for dtype, dtype_min, dtype_max in dtypes:
        if curr_min < dtype_min or curr_max > dtype_max:
            continue
        new_dtype = dtype
        break
    if not new_dtype:
        raise ValueError(
            "Cannot cast numpy array ({}) to bit-depth of {} bits".format(str(arr.dtype), bit_depth)
        )

    if contains_float and new_dtype not in supported_floats:
        raise TypeError(
            "Array contains float. Cannot cast to numpy array ({}) to {}".format(
                str(arr.dtype), new_dtype
            )
        )

    return arr.astype(new_dtype)


def _unpack_dicom_attr(header, attr, required=False):
    if not required:
        val = header.get(attr)
    else:
        try:
            val = header[attr]
        except KeyError:
            raise KeyError(f"Tag {attr} missing from dicom")

    if type(val) is pydicom.DataElement:
        val = val.value
    return val


def _wrap_as_tuple(x, default=None):
    """Wraps individual values as tuple."""
    if default is not None and not x:
        return default

    if isinstance(x, str) or not isinstance(x, Sequence):
        x = (x,)
    elif isinstance(x, Sequence) and not isinstance(x, tuple):
        x = tuple(x)
    return x
