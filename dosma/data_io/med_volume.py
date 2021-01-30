"""MedicalVolume

This module defines `MedicalVolume`, which is a wrapper for 3D volumes.
"""
import warnings
from copy import deepcopy
from typing import List, Sequence

import nibabel as nib
import numpy as np
import pydicom
from nibabel.spatialimages import SpatialFirstSlicer as _SpatialFirstSlicerNib

from dosma.data_io import orientation as stdo
from dosma.data_io.format_io import ImageDataFormat
from dosma.defaults import SCANNER_ORIGIN_DECIMAL_PRECISION
from dosma.utils import env

if env.sitk_available():
    import SimpleITK as sitk

__all__ = ["MedicalVolume"]


class MedicalVolume(object):
    """The class for medical images.

    Medical volumes are 3D matrices representing medical data. These volumes have inherent
    metadata, such as pixel/voxel spacing, global coordinates, rotation information, all of
    which can be characterized by an affine matrix following the RAS+ coordinate system.

    Standard math and boolean operations are supported with other ``MedicalVolume`` objects,
    numpy arrays (following standard broadcasting), and scalars. Boolean operations are performed
    elementwise, resulting in a volume with shape as ``self.volume.shape``.
    If performing operations between ``MedicalVolume`` objects, both objects must have
    the same shape and affine matrix (spacing, direction, and origin). Header information
    is not deep copied when performing these operations to reduce computational and memory
    overhead. The affine matrix (``self.affine``) is copied as it is lightweight and
    often modified.

    2D images are also supported when viewed trivial 3D volumes with shape ``(H, W, 1)``.

    Many operations are in-place and modify the instance directly (e.g. `reformat(inplace=True)`).
    To allow chaining operations, operations that are in-place return ``self``.

    Args:
        volume (np.ndarray): 3D volume.
        affine (np.ndarray): 4x4 array corresponding to affine matrix transform in RAS+ coordinates.
        headers (list[pydicom.FileDataset]): Headers for DICOM files.
    """

    def __init__(
        self, volume: np.ndarray, affine: np.ndarray, headers: List[pydicom.FileDataset] = None
    ):
        if headers and len(headers) != volume.shape[-1]:
            raise ValueError(
                "Header mismatch. {:d} headers, but {:d} slices".format(
                    len(headers), volume.shape[-1]
                )
            )

        self._volume = volume
        self._affine = np.array(affine)
        self._headers = headers

    def save_volume(self, file_path: str, data_format: ImageDataFormat = ImageDataFormat.nifti):
        """Write volumes in specified data format.

        Args:
            file_path (str): File path to save data. May be modified to follow convention
                given by the data format in which the volume will be saved.
            data_format (ImageDataFormat): Format to save data.
        """
        import dosma.data_io.format_io_utils

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
        new_orientation = tuple(new_orientation)
        if new_orientation == self.orientation:
            if inplace:
                return self
            return self._partial_clone(volume=self._volume)

        temp_orientation = self.orientation
        temp_affine = np.array(self._affine)

        transpose_inds = stdo.get_transpose_inds(temp_orientation, new_orientation)
        all_transpose_inds = transpose_inds + tuple(range(3, self._volume.ndim))

        volume = np.transpose(self.volume, all_transpose_inds)
        for i in range(len(transpose_inds)):
            temp_affine[..., i] = self._affine[..., transpose_inds[i]]

        temp_orientation = tuple([self.orientation[i] for i in transpose_inds])

        flip_axs_inds = list(stdo.get_flip_inds(temp_orientation, new_orientation))

        volume = np.flip(volume, axis=flip_axs_inds)

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
            mv = self
        else:
            mv = self._partial_clone(volume=volume, affine=temp_affine)

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

        return self.is_same_dimensions(mv) and (mv.volume == self.volume).all()

    def __allclose_spacing(self, mv, precision: int = None):
        """Check if spacing between self and another medical volume is within tolerance.

        Tolerance is `10 ** (-precision)`.

        Args:
            mv (MedicalVolume): Volume to compare with.
            precision (`int`, optional): Number of significant figures after the decimal.
                If not specified, check that affine matrices between two volumes are identical.
                Defaults to `None`.

        Returns:
            bool: `True` if spacing between two volumes within tolerance, `False` otherwise.
        """
        if precision:
            tol = 10 ** (-precision)
            return np.allclose(mv.affine[:3, :3], self.affine[:3, :3], atol=tol) and np.allclose(
                mv.scanner_origin, self.scanner_origin, rtol=tol
            )
        else:
            return (mv.affine == self.affine).all()

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

        is_close_spacing = self.__allclose_spacing(mv, precision)
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

    def to_sitk(self, vdim: int = None):
        """Converts to SimpleITK Image.

        SimpleITK Image objects support vector pixel types, which are represented
        as an extra dimension in numpy arrays. The vector dimension can be specified
        with ``vdim``.

        Args:
            vdim (int, optional): The vector dimension.

        Note:
            Header information is not currently copied.

        Returns:
            SimpleITK.Image
        """
        if not env.sitk_available():
            raise ImportError("SimpleITK is not installed. Install it with `pip install simpleitk`")

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

    @property
    def volume(self):
        """np.ndarray: 3D numpy array representing volume values."""
        return self._volume

    @volume.setter
    def volume(self, value: np.ndarray):
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

    @property
    def headers(self):
        """list[pydicom.FileDataset]: Headers for DICOM files."""
        return self._headers

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
        return MedicalVolume(**kwargs)

    def __getitem__(self, _slice):
        slicer = _SpatialFirstSlicer(self)
        try:
            _slice = slicer.check_slicing(_slice)
        except ValueError as err:
            raise IndexError(*err.args)

        volume = self._volume[_slice]
        if any(dim == 0 for dim in volume.shape):
            raise IndexError("Empty slice requested")

        affine = slicer.slice_affine(_slice)
        # slicing data makes headers invalid
        return self._partial_clone(volume=volume, affine=affine, headers=None)

    def __setitem__(self, _slice, value):
        if isinstance(value, MedicalVolume):
            image = self[_slice]
            assert value.is_same_dimensions(image, err=True)
            value = value._volume
        self._volume[_slice] = value

    def __add__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__add__(other)
        return self._partial_clone(volume=volume)

    def __floordiv__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__floordiv__(other)
        return self._partial_clone(volume=volume)

    def __mul__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__mul__(other)
        return self._partial_clone(volume=volume)

    def __pow__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__pow__(other)
        return self._partial_clone(volume=volume)

    def __sub__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__sub__(other)
        return self._partial_clone(volume=volume)

    def __truediv__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = self._volume.__truediv__(other)
        return self._partial_clone(volume=volume)

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

    def __ne__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume != other).astype(np.uint8)
        return self._partial_clone(volume=volume)

    def __eq__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume == other).astype(np.uint8)
        return self._partial_clone(volume=volume)

    def __ge__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume >= other).astype(np.uint8)
        return self._partial_clone(volume=volume)

    def __gt__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume > other).astype(np.uint8)
        return self._partial_clone(volume=volume)

    def __le__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume <= other).astype(np.uint8)
        return self._partial_clone(volume=volume)

    def __lt__(self, other):
        if isinstance(other, MedicalVolume):
            assert self.is_same_dimensions(other, err=True)
            other = other.volume
        volume = (self._volume < other).astype(np.uint8)
        return self._partial_clone(volume=volume)


class _SpatialFirstSlicer(_SpatialFirstSlicerNib):
    def __init__(self, img):
        self.img = img

    def __getitem__(self, slicer):
        raise NotImplementedError("Slicing should be done by `MedicalVolume`")
