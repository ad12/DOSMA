"""MedicalVolume

This module defines `MedicalVolume`, which is a wrapper for 3D volumes.
"""
import warnings
from copy import deepcopy
from numbers import Number
from typing import Sequence, Tuple, Union

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialFirstSlicer as _SpatialFirstSlicerNib
from numpy.lib.mixins import NDArrayOperatorsMixin

from dosma.data_io import orientation as stdo
from dosma.data_io.format_io import ImageDataFormat
from dosma.defaults import SCANNER_ORIGIN_DECIMAL_PRECISION
from dosma.utils import env
from dosma.utils.device import Device, cpu_device, get_array_module, get_device, to_device

if env.sitk_available():
    import SimpleITK as sitk
if env.cupy_available():
    import cupy as cp
if env.package_available("h5py"):
    import h5py

__all__ = ["MedicalVolume"]


_HANDLED_NUMPY_FUNCTIONS = {}


class MedicalVolume(NDArrayOperatorsMixin):
    """The class for medical images.

    Medical volumes use ndarrays to represent medical data. However, unlike standard ndarrays,
    these volumes have inherent spatial metadata, such as pixel/voxel spacing, global coordinates,
    rotation information, all of which can be characterized by an affine matrix following the
    RAS+ coordinate system. The code below creates a random 300x300x40 medical volume with
    spatial origin ``(0, 0, 0)`` and voxel spacing of ``(1,1,1)``:

    >>> mv = MedicalVolume(np.random.rand(300, 300, 40), np.eye(4))

    Medical volumes can also store header information that accompanies pixel data
    (e.g. DICOM headers). These headers are used to expose metadata, which can be fetched
    and set using :meth:`get_metadata()` and :meth:`set_metadata()`, respectively. Headers are
    also auto-aligned, which means that headers will be aligned with the slice(s) of data from
    which they originated, which makes Python slicing feasible. Currently, medical volumes
    support DICOM headers using ``pydicom`` when loaded with :class:``dosma.data_io.DicomReader``.

    >>> mv.get_metadata("EchoTime")  # Returns EchoTime
    >>> mv.set_metadata("EchoTime", 10.0)  # Sets EchoTime to 10.0

    Standard math and boolean operations are supported with other ``MedicalVolume`` objects,
    numpy arrays (following standard broadcasting), and scalars. Boolean operations are performed
    elementwise, resulting in a volume with shape as ``self.volume.shape``.
    If performing operations between ``MedicalVolume`` objects, both objects must have
    the same shape and affine matrix (spacing, direction, and origin). Header information
    is not deep copied when performing these operations to reduce computational and memory
    overhead. The affine matrix (``self.affine``) is copied as it is lightweight and
    often modified.

    2D images are also supported when viewed trivial 3D volumes with shape ``(H, W, 1)``:

    >>> mv = MedicalVolume(np.random.rand(10,20,1), np.eye(4))

    Many operations are in-place and modify the instance directly (e.g. `reformat(inplace=True)`).
    To allow chaining operations, operations that are in-place return ``self``.

    >>> mv2 = mv.reformat(ornt, inplace=True)
    >>> id(mv2) == id(mv)
    True

    **BETA**: Medical volumes can interface with the gpu using the :mod:`cupy` library.
    Volumes can be moved between devices (see ``dosma.Device``) using the ``.to()`` method.
    Only the volume data will be moved to the gpu. Headers and affine matrix will remain on
    the cpu. The following code moves a MedicalVolume to gpu 0 and back to the cpu:

    >>> from dosma import Device
    >>> mv = MedicalVolume(np.random.rand((10,20,30)), np.eye(4))
    >>> mv_gpu = mv.to(Device(0))
    >>> mv_cpu = mv.cpu()

    Note, moving data across devices results in a full copy. Above, ``mv_cpu.volume`` and
    ``mv.volume`` do not share memory. Saving volumes and converting to other images
    (e.g. ``SimpleITK.Image``) are only supported for cpu volumes. Volumes can also only
    be compared when on the same device. For example, both commands below will raise a
    RuntimeError:

    >>> mv_gpu == mv_cpu
    >>> mv_gpu.is_identical(mv_cpu)

    While CuPy requires the current device be set using ``cp.cuda.Device(X).use()`` or inside
    the ``with`` context, ``MedicalVolume`` automatically sets the appropriate context
    for performing operations. This means the CuPy current device need to be the same as the
    ``MedicalVolume`` object. For example, the following still works:

    >>> cp.cuda.Device(0).use()
    >>> mv_gpu = MedicalVolume(cp.ones((3,3,3)), np.eye(4))
    >>> cp.cuda.Device(1).use()
    >>> mv_gpu *= 2

    **BETA**: MedicalVolumes also have a limited NumPy/CuPy-compatible interface.
    Standard numpy/cupy functions that preserve array shapes can be performed
    on MedicalVolume objects:

    >>> log_arr = np.log(mv)
    >>> type(log_arr)
    <class 'dosma.data_io.MedicalVolume'>
    >>> exp_arr_gpu = cp.exp(mv_gpu)
    >>> type(exp_arr_gpu)
    <class 'dosma.data_io.MedicalVolume'>

    Args:
        volume (array-like): nD medical image.
        affine (array-like): 4x4 array corresponding to affine matrix transform in RAS+ coordinates.
            Must be on cpu (i.e. no ``cupy.ndarray``).
        headers (array-like[pydicom.FileDataset]): Headers for DICOM files.
    """

    def __init__(self, volume, affine, headers=None):
        xp = get_array_module(volume)
        self._volume = xp.asarray(volume)
        self._affine = np.array(affine)
        self._headers = self._validate_and_format_headers(headers) if headers is not None else None

    def save_volume(self, file_path: str, data_format: ImageDataFormat = ImageDataFormat.nifti):
        """Write volumes in specified data format.

        Args:
            file_path (str): File path to save data. May be modified to follow convention
                given by the data format in which the volume will be saved.
            data_format (ImageDataFormat): Format to save data.
        """
        import dosma.data_io.format_io_utils

        device = self.device
        if device != cpu_device:
            raise RuntimeError(f"MedicalVolume must be on cpu, got {self.device}")

        writer = dosma.data_io.format_io_utils.get_writer(data_format)
        writer.save(self, file_path)

    def reformat(self, new_orientation: Sequence, inplace: bool = False) -> "MedicalVolume":
        """Reorients volume to a specified orientation.

        Flipping and transposing the volume array (``self.volume``) returns a view if possible.

        Reorientation method:
        ---------------------
        - Axis transpose and flipping are linear operations and therefore can be treated
        independently.
        - working example: ('AP', 'SI', 'LR') --> ('RL', 'PA', 'SI')
        1. Transpose volume and RAS orientation to appropriate column in matrix
            eg. ('AP', 'SI', 'LR') --> ('LR', 'AP', 'SI') - transpose_inds=[2, 0, 1]
        2. Flip volume across corresponding axes
            eg. ('LR', 'AP', 'SI') --> ('RL', 'PA', 'SI') - flip axes 0,1

        Reorientation method implementation:
        ------------------------------------
        1. Transpose: Switching (transposing) axes in volume is the same as switching columns
        in affine matrix

        2. Flipping: Negate each column corresponding to pixel axis to flip (i, j, k) and
        reestablish origins based on flipped axes

        Args:
            new_orientation (Sequence): New orientation.
            inplace (bool, optional): If `True`, do operation in-place and return ``self``.

        Returns:
            MedicalVolume: The reformatted volume. If ``inplace=True``, returns ``self``.
        """
        xp = self.device.xp
        device = self.device
        headers = self._headers

        new_orientation = tuple(new_orientation)
        if new_orientation == self.orientation:
            if inplace:
                return self
            return self._partial_clone(volume=self._volume)

        temp_orientation = self.orientation
        temp_affine = np.array(self._affine)

        transpose_inds = stdo.get_transpose_inds(temp_orientation, new_orientation)
        all_transpose_inds = transpose_inds + tuple(range(3, self._volume.ndim))

        with device:
            volume = xp.transpose(self.volume, all_transpose_inds)
        if headers is not None:
            headers = np.transpose(headers, all_transpose_inds)
        for i in range(len(transpose_inds)):
            temp_affine[..., i] = self._affine[..., transpose_inds[i]]

        temp_orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = list(stdo.get_flip_inds(temp_orientation, new_orientation))
        with device:
            volume = xp.flip(volume, axis=tuple(flip_axs_inds))
        if headers is not None:
            headers = np.flip(headers, axis=tuple(flip_axs_inds))
        a_vecs = temp_affine[:3, :3]
        a_origin = temp_affine[:3, 3]

        # phi is a vector of 1s and -1s, where 1 indicates no flip, and -1 indicates flip
        # phi is used to determine which columns in affine matrix to flip
        phi = np.ones([1, len(a_origin)]).flatten()
        phi[flip_axs_inds] *= -1

        b_vecs = np.array(a_vecs)
        for i in range(len(phi)):
            b_vecs[:, i] *= phi[i]

        # get number of pixels to shift by on each axis.
        # Should be 0 when not flipping - i.e. phi<0 mask
        vol_shape_vec = (
            (np.asarray(volume.shape[:3]) - 1) * (phi < 0).astype(np.float32)
        ).transpose()
        b_origin = np.round(
            a_origin.flatten() - np.matmul(b_vecs, vol_shape_vec).flatten(),
            SCANNER_ORIGIN_DECIMAL_PRECISION,
        )

        temp_affine = np.array(self.affine)
        temp_affine[:3, :3] = b_vecs
        temp_affine[:3, 3] = b_origin
        temp_affine[temp_affine == 0] = 0  # get rid of negative 0s

        if inplace:
            self._affine = temp_affine
            self._volume = volume
            self._headers = headers
            mv = self
        else:
            mv = self._partial_clone(volume=volume, affine=temp_affine, headers=headers)

        assert (
            mv.orientation == new_orientation
        ), f"Orientation mismatch: Expected: {self.orientation}. Got {new_orientation}"
        return mv

    def reformat_as(self, other, inplace: bool = False) -> "MedicalVolume":
        """Reformat this to the same orientation as ``other``.
        Equivalent to ``self.reformat(other.orientation, inplace)``.

        Args:
            other (MedicalVolume): The result volume has the same orientation as ``other``.
            inplace (bool, optional): If `True`, do operation in-place and return ``self``.

        Returns:
            MedicalVolume: The reformatted volume. If ``inplace=True``, returns ``self``.
        """
        return self.reformat(other.orientation, inplace=inplace)

    def is_identical(self, mv):
        """Check if another medical volume is identical.

        Two volumes are identical if they have the same pixel_spacing, orientation,
        scanner_origin, and volume.

        Args:
            mv (MedicalVolume): Volume to compare with.

        Returns:
            bool: `True` if identical, `False` otherwise.
        """
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        idevice = self.device
        odevice = mv.device
        if idevice != odevice:
            raise RuntimeError(f"Expected device {idevice}, got {odevice}.")

        with idevice:
            return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def _allclose_spacing(self, mv, precision: int = None, ignore_origin: bool = False):
        """Check if spacing between self and another medical volume is within tolerance.

        Tolerance is `10 ** (-precision)`.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal.
                If not specified, check that affine matrices between two volumes are identical.
                Defaults to `None`.
            ignore_origin (bool, optional): If ``True``, ignore matching origin in the affine
                matrix.

        Returns:
            bool: `True` if spacing between two volumes within tolerance, `False` otherwise.
        """
        if precision is not None:
            tol = 10 ** (-precision)
            return np.allclose(mv.affine[:3, :3], self.affine[:3, :3], atol=tol) and (
                ignore_origin or np.allclose(mv.scanner_origin, self.scanner_origin, rtol=tol)
            )
        else:
            return (mv.affine == self.affine).all() or (
                ignore_origin and (mv.affine[:, :3] == self.affine[:, :3]).all()
            )

    def is_same_dimensions(self, mv, precision: int = None, err: bool = False):
        """Check if two volumes have the same dimensions.

        Two volumes have the same dimensions if they have the same pixel_spacing,
        orientation, and scanner_origin.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal.
                If not specified, check that affine matrices between two volumes are identical.
                Defaults to `None`.
            err (bool, optional): If `True` and volumes do not have same dimensions,
                raise descriptive ValueError.

        Returns:
            bool: ``True`` if pixel spacing, orientation, and scanner origin
                between two volumes within tolerance, ``False`` otherwise.

        Raises:
            TypeError: If ``mv`` is not a MedicalVolume.
            ValueError: If ``err=True`` and two volumes do not have same dimensions.
        """
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        is_close_spacing = self._allclose_spacing(mv, precision)
        is_same_orientation = mv.orientation == self.orientation
        is_same_shape = mv.volume.shape == self.volume.shape
        out = is_close_spacing and is_same_orientation and is_same_shape

        if err and not out:
            tol_str = f" (tol: 1e-{precision})" if precision else ""
            if not is_close_spacing:
                raise ValueError(
                    "Affine matrices not equal{}:\n{}\n{}".format(tol_str, self._affine, mv._affine)
                )
            if not is_same_orientation:
                raise ValueError(
                    "Orientations not equal: {}, {}".format(self.orientation, mv.orientation)
                )
            if not is_same_shape:
                raise ValueError(
                    "Shapes not equal: {}, {}".format(self._volume.shape, mv._volume.shape)
                )
            assert False  # should not reach here

        return out

    def match_orientation(self, mv):
        """Reorient another MedicalVolume to orientation specified by self.orientation.

        Args:
            mv (MedicalVolume): Volume to reorient.
        """
        warnings.warn(
            "`match_orientation` is deprecated and will be removed in v0.1. "
            "Use `mv.reformat_as(self, inplace=True)` instead.",
            DeprecationWarning,
        )
        if not isinstance(mv, MedicalVolume):
            raise TypeError("`mv` must be a MedicalVolume.")

        mv.reformat(self.orientation, inplace=True)

    def match_orientation_batch(self, mvs):
        """Reorient a collection of MedicalVolumes to orientation specified by self.orientation.

        Args:
            mvs (list[MedicalVolume]): Collection of MedicalVolumes.
        """
        warnings.warn(
            "`match_orientation_batch` is deprecated and will be removed in v0.1. "
            "Use `[x.reformat_as(self, inplace=True) for x in mvs]` instead.",
            DeprecationWarning,
        )
        for mv in mvs:
            self.match_orientation(mv)

    def clone(self, headers=True):
        """Clones the medical volume.

        Args:
            headers (bool, optional): If `True`, clone headers.
                If `False`, headers have shared memory.

        Returns:
            mv (MedicalVolume): A cloned MedicalVolume.
        """
        return MedicalVolume(
            self.volume.copy(),
            self.affine.copy(),
            headers=deepcopy(self._headers) if headers else self._headers,
        )

    def to(self, device):
        """Move to device.

        If on same device, no-op and returns ``self``.

        Args:
            device: The device to move to.

        Returns:
            MedicalVolume
        """
        device = Device(device)
        if self.device == device:
            return self

        return self._partial_clone(volume=to_device(self._volume, device))

    def cpu(self):
        """Move to cpu."""
        return self.to("cpu")

    def astype(self, dtype, **kwargs):
        """Modifies dtype of ``self._volume``.

        Note this operation is done in place. ``self._volume`` is modified, based
        on the ``astype`` implementation of the type associated with ``self._volume``.
        No new MedicalVolume is created - ``self`` is returned.

        Args:
            dtype (str or dtype): Typecode or data-type to which the array is cast.

        Returns:
            self
        """
        if (
            env.package_available("h5py")
            and isinstance(self._volume, h5py.Dataset)
            and env.get_version(h5py) < (3, 0, 0)
        ):
            raise ValueError("Cannot cast h5py.Dataset to dtype for h5py<3.0.0")

        self._volume = self._volume.astype(dtype, **kwargs)
        return self

    def to_sitk(self, vdim: int = None):
        """Converts to SimpleITK Image.

        SimpleITK Image objects support vector pixel types, which are represented
        as an extra dimension in numpy arrays. The vector dimension can be specified
        with ``vdim``.

        MedicalVolume must be on cpu. Use ``self.cpu()`` to move.

        Args:
            vdim (int, optional): The vector dimension.

        Note:
            Header information is not currently copied.

        Returns:
            SimpleITK.Image

        Raises:
            ImportError: If `SimpleITK` is not installed.
            RuntimeError: If MedicalVolume is not on cpu.

        Note:
            Header information is not currently copied.
        """
        if not env.sitk_available():
            raise ImportError("SimpleITK is not installed. Install it with `pip install simpleitk`")
        device = self.device
        if device != cpu_device:
            raise RuntimeError(f"MedicalVolume must be on cpu, got {self.device}")

        arr = self.volume
        ndim = arr.ndim

        if vdim is not None:
            if vdim < 0:
                vdim = ndim + vdim
            axes = tuple(i for i in range(ndim) if i != vdim)[::-1] + (vdim,)
        else:
            axes = range(ndim)[::-1]
        arr = np.transpose(arr, axes)

        affine = self.affine.copy()
        affine[:2] = -affine[:2]  # RAS+ -> LPS+

        origin = tuple(affine[:3, 3])
        spacing = self.pixel_spacing
        direction = affine[:3, :3] / np.asarray(spacing)

        img = sitk.GetImageFromArray(arr, isVector=vdim is not None)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(tuple(direction.flatten()))

        return img

    def headers(self, flatten=False):
        """Returns headers"""
        if flatten:
            return self._headers.flatten()
        return self._headers

    def get_metadata(self, key, dtype=None):
        """Get metadata value from first header.

        The first header is defined as the first header in ``np.flatten(self._headers)``.
        To extract header information for other headers, use ``self.headers()``.

        Args:
            key (``str`` or pydicom.BaseTag``): Metadata field to access.
            dtype (type, optional): If specified, data type to cast value to.
                By default for DICOM headers, data will be in the value
                representation format specified by pydicom. See
                ``pydicom.valuerep``.

        Examples:
            >>> mv.get_metadata("EchoTime")
            '10.0'  # this is a number type ``pydicom.valuerep.DSDecimal``
            >>> mv.get_metadata("EchoTime", dtype=float)
            10.0

        Note:
            Currently header information is tied to the ``pydicom.FileDataset`` implementation.
            This function is synonymous to ``dataset.<key>`` in ``pydicom.FileDataset``.
        """
        if self._headers is None:
            raise RuntimeError("No headers found. MedicalVolume must be initialized with `headers`")
        headers = self.headers(flatten=True)
        element = headers[0][key]
        val = element.value
        if dtype is not None:
            val = dtype(val)
        return val

    def set_metadata(self, key, value, force: bool = False):
        """Sets metadata for all headers.

        Args:
            key (str or pydicom.BaseTag): Metadata field to access.
            value (Any): The value.
            force (bool, optional): If ``True``, force the header to
                set key even if key does not exist in header.
        """
        if self._headers is None:
            raise RuntimeError("No headers found. MedicalVolume must be initialized with `headers`")
        for h in self.headers(flatten=True):
            if force:
                setattr(h, key, value)
            else:
                h[key].value = value

    def round(self, decimals=0, affine=False):
        return around(self, decimals, affine)

    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=np._NoValue,
        where=np._NoValue,
    ):
        """Identical to :method:``sum_np``."""
        # `out` is required for cupy arrays because of how cupy calls array.
        if out is not None:
            raise ValueError("`out` must be None")
        return sum_np(self, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, where=np._NoValue):
        """Identical to :method:``mean_np``."""
        # `out` is required for cupy arrays because of how cupy calls array.
        if out is not None:
            raise ValueError("`out` must be None")
        return mean_np(self, axis=axis, dtype=dtype, keepdims=keepdims, where=where)

    @property
    def volume(self):
        """ndarray: 3D ndarray representing volume values."""
        return self._volume

    @volume.setter
    def volume(self, value):
        """
        If the volume is of a different shape, the headers are no longer valid,
        so delete all reorientations are done as part of MedicalVolume,
        so reorientations are permitted.

        However, external setting of the volume to a different shape array is not allowed.
        """
        if value.ndim != self._volume.ndim:
            raise ValueError("New volume must be same as current volume")

        if self._volume.shape != value.shape:
            self._headers = None

        self._volume = value
        self._device = get_device(self._volume)

    @property
    def pixel_spacing(self):
        """tuple[float]: Pixel spacing in order of current orientation."""
        vecs = self._affine[:3, :3]
        ps = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

        assert len(ps) == 3, "Pixel spacing must have length of 3"
        return ps

    @property
    def orientation(self):
        """tuple[str]: Image orientation in standard orientation format.

        See orientation.py for more information on conventions.
        """
        nib_orientation = nib.aff2axcodes(self._affine)
        return stdo.orientation_nib_to_standard(nib_orientation)

    @property
    def scanner_origin(self):
        """tuple[float]: Scanner origin in global RAS+ x,y,z coordinates.
        """
        return tuple(self._affine[:3, 3])

    @property
    def affine(self):
        """np.ndarray: 4x4 affine matrix for volume in current orientation."""
        return self._affine

    @property
    def shape(self):
        return self._volume.shape

    @property
    def device(self):
        return get_device(self._volume)

    @property
    def dtype(self):
        return self._volume.dtype

    @classmethod
    def from_sitk(cls, image, copy=False) -> "MedicalVolume":
        """Constructs MedicalVolume from SimpleITK.Image.

        Note:
            Metadata information is not copied.

        Args:
            image (SimpleITK.Image): The image.
            copy (bool, optional): If ``True``, copies array.

        Returns:
            MedicalVolume
        """
        if not env.sitk_available():
            raise ImportError("SimpleITK is not installed. Install it with `pip install simpleitk`")

        if len(image.GetSize()) < 3:
            raise ValueError("`image` must be 3D.")
        is_vector_image = image.GetNumberOfComponentsPerPixel() > 1

        if copy:
            arr = sitk.GetArrayFromImage(image)
        else:
            arr = sitk.GetArrayViewFromImage(image)

        ndim = arr.ndim
        if is_vector_image:
            axes = tuple(range(ndim)[-2::-1]) + (ndim - 1,)
        else:
            axes = range(ndim)[::-1]
        arr = np.transpose(arr, axes)

        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = np.asarray(image.GetDirection()).reshape(-1, 3)

        affine = np.zeros((4, 4))
        affine[:3, :3] = direction * np.asarray(spacing)
        affine[:3, 3] = origin
        affine[:2] = -affine[:2]  # LPS+ -> RAS+
        affine[3, 3] = 1

        return cls(arr, affine)

    def _partial_clone(self, **kwargs) -> "MedicalVolume":
        """Copies constructor information from ``self`` if not available in ``kwargs``."""
        for k in ("volume", "affine"):
            if k not in kwargs:
                kwargs[k] = getattr(self, f"_{k}").copy()
        if "headers" not in kwargs:
            kwargs["headers"] = self._headers
        elif isinstance(kwargs["headers"], bool) and kwargs["headers"]:
            kwargs["headers"] = deepcopy(self._headers)
        return self.__class__(**kwargs)

    def _validate_and_format_headers(self, headers):
        """Validate headers are of appropriate shape and format into standardized shape.

        Headers are stored an ndarray of dictionary-like objects with explicit dimensions
        that match the dimensions of ``self._volume``. If header objects are not

        Assumes ``self._volume`` and ``self._affine`` have been set.
        """
        headers = np.asarray(headers)
        if headers.ndim > self._volume.ndim:
            raise ValueError(
                f"`headers` has too many dimensions. "
                f"Got headers.ndim={headers.ndim}, but volume.ndim={self._volume.ndim}"
            )
        for dim in range(-headers.ndim, 0)[::-1]:
            if headers.shape[dim] not in (1, self._volume.shape[dim]):
                raise ValueError(
                    f"`headers` must follow standard broadcasting shape. "
                    f"Got headers.shape={headers.shape}, but volume.shape={self._volume.shape}"
                )

        ndim = self._volume.ndim
        shape = (1,) * (ndim - len(headers.shape)) + headers.shape
        headers = np.reshape(headers, shape)
        return headers

    def _extract_input_array_ufunc(self, input, device=None):
        if device is None:
            device = self.device
        device_err = "Expected device {} but got device ".format(device) + "{}"
        if isinstance(input, Number):
            return input
        elif isinstance(input, np.ndarray):
            if device != cpu_device:
                raise RuntimeError(device_err.format(cpu_device))
            return input
        elif env.cupy_available() and isinstance(input, cp.ndarray):
            if device != input.device:
                raise RuntimeError(device_err.format(Device(input.device)))
            return input
        elif isinstance(input, MedicalVolume):
            if device != input.device:
                raise RuntimeError(device_err.format(Device(input.device)))
            assert self.is_same_dimensions(input, err=True)
            return input._volume
        else:
            return NotImplemented

    def _check_reduce_axis(self, axis: Union[int, Sequence[int]]) -> Tuple[int]:
        if axis is None:
            return None
        is_sequence = isinstance(axis, Sequence)
        if not is_sequence:
            axis = (axis,)
        axis = tuple(x if x >= 0 else self.volume.ndim + x for x in axis)
        assert all(x >= 0 for x in axis)
        if any(x < 3 for x in axis):
            raise ValueError("Cannot reduce MedicalVolume along spatial dimensions")
        if not is_sequence:
            axis = axis[0]
        return axis

    def _reduce_array(self, func, *inputs, **kwargs) -> "MedicalVolume":
        """
        Assumes inputs have been verified.
        """
        device = self.device
        xp = device.xp

        keepdims = kwargs.get("keepdims", False)
        reduce_axis = self._check_reduce_axis(kwargs["axis"])
        kwargs["axis"] = reduce_axis
        if not isinstance(reduce_axis, Sequence):
            reduce_axis = (reduce_axis,)
        with device:
            volume = func(*inputs, **kwargs)

        if xp.isscalar(volume) or volume.ndim == 0:
            return volume

        if self._headers is not None:
            headers_slices = tuple(
                slice(None) if x not in reduce_axis else slice(0, 1) if keepdims else 0
                for x in range(self._headers.ndim)
            )
            headers = self._headers[headers_slices]
        else:
            headers = None
        return self._partial_clone(volume=volume, headers=headers)

    def __getitem__(self, _slice):
        slicer = _SpatialFirstSlicer(self)
        try:
            _slice = slicer.check_slicing(_slice)
        except ValueError as err:
            raise IndexError(*err.args)

        volume = self._volume[_slice]
        if any(dim == 0 for dim in volume.shape):
            raise IndexError("Empty slice requested")

        headers = self._headers
        if headers is not None:
            _slice_headers = []
            for idx, x in enumerate(_slice):
                if headers.shape[idx] == 1 and not isinstance(x, int):
                    _slice_headers.append(slice(None))
                elif headers.shape[idx] == 1 and isinstance(x, int):
                    _slice_headers.append(0)
                else:
                    _slice_headers.append(x)
            headers = headers[_slice_headers]

        affine = slicer.slice_affine(_slice)
        return self._partial_clone(volume=volume, affine=affine, headers=headers)

    def __setitem__(self, _slice, value):
        """
        Note:
            When ``value`` is a ``MedicalVolume``, the headers from that value
            are not copied over. This may be changed in the future.
        """
        if isinstance(value, MedicalVolume):
            image = self[_slice]
            assert value.is_same_dimensions(image, err=True)
            value = value._volume
        with self.device:
            self._volume[_slice] = value

    def __repr__(self) -> str:
        nl = "\n"
        return f"{self.__class__.__name__}(volume={self._volume},{nl}affine={self._affine})"

    def __iadd__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__iadd__(other)
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__ifloordiv__(other)
        return self

    def __imul__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__imul__(other)
        return self

    def __ipow__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__ipow__(other)
        return self

    def __isub__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__isub__(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        self._volume.__itruediv__(other)
        return self

    def __array__(self):
        """Wrapper for performing numpy operations on MedicalVolume array.

        Examples:
            >>> a = np.asarray(mv)
            >>> type(a)
            <class 'numpy.ndarray'>

        Note:
            This is not valid when ``self.volume`` is a ``cupy.ndarray``.
            All CUDA ndarrays must first be moved to the cpu.
        """
        try:
            return np.asarray(self.volume)
        except TypeError:
            raise TypeError(
                "Implicit conversion to a NumPy array is not allowed. "
                "Please use `.cpu()` to move the array to the cpu explicitly "
                "before constructing a NumPy array."
            )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        def _extract_inputs(inputs, device):
            _inputs = []
            for input in inputs:
                input = self._extract_input_array_ufunc(input, device)
                if input is NotImplemented:
                    return input
                _inputs.append(input)
            return _inputs

        if method not in ["__call__", "reduce"]:
            return NotImplemented

        device = self.device
        _inputs = _extract_inputs(inputs, device)
        if _inputs is NotImplemented:
            return NotImplemented

        if method == "__call__":
            with device:
                volume = ufunc(*_inputs, **kwargs)
            if volume.shape != self._volume.shape:
                raise ValueError(
                    f"{self.__class__.__name__} does not support operations that change shape. "
                    f"Use operations on `self.volume` to modify array objects."
                )
            return self._partial_clone(volume=volume)
        elif method == "reduce":
            return self._reduce_array(ufunc.reduce, *_inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _HANDLED_NUMPY_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MedicalVolume objects.
        if not all(issubclass(t, (MedicalVolume, self.__class__)) for t in types):
            return NotImplemented
        return _HANDLED_NUMPY_FUNCTIONS[func](*args, **kwargs)

    @property
    def __cuda_array_interface__(self):
        """Wrapper for performing cupy operations on MedicalVolume array.
        """
        if self.device == cpu_device:
            raise TypeError(
                "Implicit conversion to a CuPy array is not allowed. "
                "Please use `.to(device)` to move the array to the gpu explicitly "
                "before constructing a CuPy array."
            )
        return self.volume.__cuda_array_interface__


class _SpatialFirstSlicer(_SpatialFirstSlicerNib):
    def __init__(self, img):
        self.img = img

    def __getitem__(self, slicer):
        raise NotImplementedError("Slicing should be done by `MedicalVolume`")


# =================================
# Supported numpy functions
# =================================
def implements(*np_functions):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        for np_func in np_functions:
            _HANDLED_NUMPY_FUNCTIONS[np_func] = func
        return func

    return decorator


def reduce_array_op(func, x, axis=None, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v != np._NoValue}
    input = x._extract_input_array_ufunc(x)
    if input is NotImplemented:
        return NotImplemented
    return x._reduce_array(func, input, axis=axis, **kwargs)


@implements(np.amin)
def amin(x, axis=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    return reduce_array_op(np.amin, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@implements(np.amax)
def amax(x, axis=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    return reduce_array_op(np.amax, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@implements(np.argmin)
def argmin(x, axis=None):
    return reduce_array_op(np.argmin, x, axis=axis)


@implements(np.argmax)
def argmax(x, axis=None):
    return reduce_array_op(np.argmax, x, axis=axis)


@implements(np.sum)
def sum_np(x, axis=None, dtype=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    return reduce_array_op(
        np.sum, x, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where
    )


@implements(np.mean)
def mean_np(x, axis=None, dtype=None, keepdims=False, where=np._NoValue):
    return reduce_array_op(np.mean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@implements(np.std)
def std(x, axis=None, dtype=None, ddof=0, keepdims=False, where=np._NoValue):
    return reduce_array_op(
        np.std, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where
    )


@implements(np.nanmin)
def nanmin(x, axis=None, keepdims=False):
    return reduce_array_op(np.nanmin, x, axis=axis, keepdims=keepdims)


@implements(np.nanmax)
def nanmax(x, axis=None, keepdims=False):
    return reduce_array_op(np.nanmax, x, axis=axis, keepdims=keepdims)


@implements(np.nanargmin)
def nanargmin(x, axis=None):
    return reduce_array_op(np.nanargmin, x, axis=axis)


@implements(np.nanargmax)
def nanargmax(x, axis=None):
    return reduce_array_op(np.nanargmax, x, axis=axis)


@implements(np.nansum)
def nansum(x, axis=None, dtype=None, keepdims=False):
    return reduce_array_op(np.nansum, x, axis=axis, dtype=dtype, keepdims=keepdims)


@implements(np.nanmean)
def nanmean(x, axis=None, dtype=None, keepdims=False):
    return reduce_array_op(np.nanmean, x, axis=axis, dtype=dtype, keepdims=keepdims)


@implements(np.nanstd)
def nanstd(x, axis=None, dtype=None, ddof=0, keepdims=False):
    return reduce_array_op(np.nanstd, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


@implements(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    vol = np.nan_to_num(x.volume, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
    if not copy:
        x._volume = vol
        return x
    else:
        return x._partial_clone(volume=vol)


@implements(np.around, np.round, np.round_)
def around(x, decimals=0, affine=False):
    """Round medical image pixel data (and optionally affine) to the given number of decimals.

    Args:
        x (MedicalVolume): A medical image.
        decimals (int, optional): Number of decimal places to round to.
            If decimals is negative, it specifies the number of positions to the left
            of the decimal point.
        affine (bool, optional): If ``True``, rounds affine matrix.
    """
    affine = np.around(x.affine, decimals=decimals) if affine else x.affine
    return x._partial_clone(volume=np.around(x.volume, decimals=decimals), affine=affine)


@implements(np.stack)
def stack(xs, axis: int = -1):
    """Stack medical images across non-spatial dimensions.

    Images will be auto-oriented to the orientation of the first medical volume.

    Args:
        xs (array-like[MedicalVolume]): 1D array-like of aligned medical images to stack.
        axis (int, optional): Axis to stack along.

    Returns:
        MedicalVolume: Stack

    Note:
        Unlike NumPy, the default stacking axis is ``-1``.

    Note:
        Headers are not set unless all inputs have headers of the same
        shape. This functionality may change in the future.
    """
    if not isinstance(axis, int):
        raise TypeError(f"'{type(axis)}' cannot be interpreted as int")

    xs = [x.reformat(xs[0].orientation) for x in xs]
    affine = xs[0].affine
    for x in xs[1:]:
        assert x.is_same_dimensions(xs[0], err=True)
    try:
        axis = _to_positive_axis(axis, len(xs[0].shape), grow=True, invalid_axis="spatial")
    except ValueError:
        raise ValueError(f"Cannot stack across spatial dimension (axis={axis})")
    assert axis >= 0

    vol = np.stack([x.volume for x in xs], axis=axis)
    headers = [x.headers() for x in xs]
    if any(x is None for x in headers):
        headers = None
    else:
        headers = np.stack(headers, axis=axis)

    return MedicalVolume(vol, affine, headers=headers)


@implements(np.concatenate)
def concatenate(xs, axis: int = -1):
    """Concatenate medical images.

    Image concatenation is slightly different if the axis is a spatial axis
    (one of the first 3 dimensions) or a non-spatial dimension.

    If concatenating along a non-spatial dimension, the image dimensions for all
    other axes and affine matrix of each ``x`` must be the same, which is standard
    for concatenation.

    If concatenating along a spatial dimension, all images must have the same direction
    and pixel spacing. Additionally, the scanner origin for all spatial axes not being
    concatenated should be the same. The origin for other scans should be consecutive.
    For example, if images are concatenated on ``axis=i``, a spatial axis, then
    ``xs[0].scanner_origin + xs[0].``.

    Images will be auto-oriented to the orientation of the first medical volume.

    Note:
        Headers are not set unless all inputs have headers of the same
        shape. This functionality may change in the future.
    """
    precision = None
    tol = 10 ** (-precision) if precision is not None else None

    if not isinstance(axis, int):
        raise TypeError(f"'{type(axis)}' cannot be interpreted as int")

    xs = [x.reformat(xs[0].orientation) for x in xs]
    axis = _to_positive_axis(axis, len(xs[0].shape), grow=False, invalid_axis=None)
    assert axis >= 0

    if axis in range(3):
        # Concatenate along spatial dimension
        for i, x in enumerate(xs[1:]):
            if not x._allclose_spacing(xs[0], precision=precision, ignore_origin=True):
                raise ValueError(
                    "All the inputs must have the same direction and pixel spacing "
                    "when concatenating spatial dimensions, but input at index 0 "
                    "has affine {} and the input at index {} "
                    "has affine {}".format(xs[0].affine[:3, :3], i, x.affine[:3, :3])
                )
        for i, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            ijk1 = np.array([0, 0, 0, 1])
            ijk1[axis] = x1.shape[axis]
            xyz = x1.affine.dot(ijk1)[:3]
            if not (
                (precision is not None and np.allclose(x2.scanner_origin, xyz, rtol=tol))
                or (np.asarray(x2.scanner_origin) == xyz).all()
            ):
                raise ValueError(
                    "All the inputs must be sequentially increasing in space "
                    "when concatenating spatial dimensions, but input at index {} "
                    "ends at xyz location {} and the input at index {} "
                    "starts at xyz location {}".format(i, xyz, i + 1, x2.scanner_origin)
                )
    else:
        for i, x in enumerate(xs[1:]):
            if not x._allclose_spacing(xs[0], precision=precision):
                raise ValueError(
                    "All the inputs must have the same affine matrix "
                    "when concatenating non-spatial dimensions, but input at index 0 "
                    "has affine {} and the input at index {} "
                    "has affine {}".format(xs[0].affine, i, x.affine)
                )

    volume = np.concatenate([x.volume for x in xs], axis=axis)
    headers = [x.headers() for x in xs]
    if any(x is None for x in headers):
        headers = None
    else:
        headers = np.concatenate(headers, axis=axis)
        if headers.ndim != volume.ndim or any(
            [hs != 1 and hs != vs for hs, vs in zip(headers.shape, volume.shape)]
        ):
            warnings.warn(
                "Got invalid headers shape ({}) given concatenated output shape ({}). "
                "Expected header dimensions to be 1 or same as volume dimension for all axes. "
                "Dropping all headers in concatenated output.".format(volume.shape, headers.shape)
            )
            headers = None

    return MedicalVolume(volume, xs[0].affine, headers=headers)


@implements(np.expand_dims)
def expand_dims(x, axis: Union[int, Sequence[int]]):
    """Expand across non-spatial dimensions.

    Args:
        x (MedicalVolume): A medical image.
        axis (``int(s)``): Axis/axes to expand dimensions.

    Returns:
        MedicalVolume: The medical image with expanded dimensions.
    """
    try:
        axis = _to_positive_axis(axis, len(x.shape), grow=True, invalid_axis="spatial")
    except ValueError:
        raise ValueError(f"Cannot expand across spatial dimensions (axis={axis})")
    vol = np.expand_dims(x.volume, axis)
    headers = x.headers()
    if headers is not None:
        headers = np.expand_dims(headers, axis)
    return x._partial_clone(volume=vol, headers=headers)


@implements(np.squeeze)
def squeeze(x, axis: Union[int, Sequence[int]] = None):
    """Squeeze non-spatial dimensions.

    Args:
        x (MedicalVolume): A medical image.
        axis (``int(s)``): Axis/axes to squeeze. Defaults to non-spatial axes.

    Returns:
        MedicalVolume: The medical image with squeezed dimensions.

    Raises:
        ValueError: If axis is not None, and an axis being squeezed is not of length 1
            or axis is not None and is squeezing spatial dimension (i.e. axis=0, 1, or 2).
    """
    if axis is not None:
        try:
            axis = _to_positive_axis(axis, len(x.shape), grow=False, invalid_axis="spatial")
        except ValueError:
            raise ValueError(f"Cannot squeeze across spatial dimensions (axis={axis})")
    else:
        axis = tuple(i for i in range(3, len(x.shape)) if x.shape[i] == 1)
        if not axis:
            return x

    vol = np.squeeze(x.volume, axis=axis)
    headers = x.headers()
    if headers is not None:
        headers = np.squeeze(headers, axis=axis)

    return x._partial_clone(volume=vol, headers=headers)


@implements(np.where)
def where(*args, **kwargs):
    return np.where(np.asarray(args[0]), *args[1:], **kwargs)


@implements(np.all)
def all_np(x, axis=None, keepdims=np._NoValue):
    return reduce_array_op(np.all, x, axis=axis, keepdims=keepdims)


@implements(np.any)
def any_np(x, axis=None, keepdims=np._NoValue):
    return reduce_array_op(np.any, x, axis=axis, keepdims=keepdims)


def _to_positive_axis(
    axis: Union[int, Sequence[int]],
    ndim: int,
    grow: bool = False,
    invalid_axis: Union[int, Sequence[int]] = None,
):
    """
    Args:
        axis (``int(s)``): The axis/axes to convert to positive.
        ndim (int): The current dimension.
        grow (bool, optional): If ``True``, converts axes to positive
            positions based on new dimension. The new dimension is
            calculated as ``ndim + #(axes < 0) + #(axes >= ndim)``.
        invalid_axis (bool, optional): Axes that are invalid.
            These should all be positive as check is done after
            positive.

    Returns:
        int(s): The positively formatted axes.
    """
    original_axis = axis

    is_sequence = isinstance(axis, Sequence)
    if not is_sequence:
        axis = (axis,)
    if grow:
        ndim += sum(tuple(x < 0 or x >= ndim for x in axis))
    axis = tuple(x if x >= 0 else ndim + x for x in axis)

    if invalid_axis is not None:
        if invalid_axis == "spatial":
            invalid_axis = tuple(range(0, 3))
        elif not isinstance(invalid_axis, Sequence):
            assert isinstance(invalid_axis, int)
            invalid_axis = (invalid_axis,)
        if any(x in invalid_axis for x in axis):
            raise ValueError(
                f"Invalid axes {original_axis}. Specified axes should not be in axes {invalid_axis}"
            )

    if not is_sequence:
        axis = axis[0]
    return axis
