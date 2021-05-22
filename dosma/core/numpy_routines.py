"""List of numpy functions supported for MedicalVolumes.
"""
import warnings
from typing import Sequence, Union

import numpy as np

from dosma.core.med_volume import MedicalVolume

__all__ = [
    "amin",
    "amax",
    "argmin",
    "argmax",
    "sum_np",
    "mean_np",
    "std",
    "nanmin",
    "nanmax",
    "nanargmin",
    "nanargmax",
    "nansum",
    "nanmean",
    "nanstd",
    "nan_to_num",
    "around",
    "clip",
    "stack",
    "concatenate",
    "expand_dims",
    "squeeze",
    "pad",
    "where",
    "all_np",
    "any_np",
    "zeros_like",
    "ones_like",
    "shares_memory",
    "may_share_memory",
]


_HANDLED_NUMPY_FUNCTIONS = {}


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
    """See :func:`numpy.amin`."""
    return reduce_array_op(np.amin, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@implements(np.amax)
def amax(x, axis=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """See :func:`numpy.amax`."""
    return reduce_array_op(np.amax, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@implements(np.argmin)
def argmin(x, axis=None):
    """See :func:`numpy.argmin`."""
    return reduce_array_op(np.argmin, x, axis=axis)


@implements(np.argmax)
def argmax(x, axis=None):
    """See :func:`numpy.argmax`."""
    return reduce_array_op(np.argmax, x, axis=axis)


@implements(np.sum)
def sum_np(x, axis=None, dtype=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """See :func:`numpy.sum`."""
    return reduce_array_op(
        np.sum, x, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where
    )


@implements(np.mean)
def mean_np(x, axis=None, dtype=None, keepdims=False, where=np._NoValue):
    """See :func:`numpy.mean`."""
    return reduce_array_op(np.mean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@implements(np.std)
def std(x, axis=None, dtype=None, ddof=0, keepdims=False, where=np._NoValue):
    """See :func:`numpy.std`."""
    return reduce_array_op(
        np.std, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where
    )


@implements(np.nanmin)
def nanmin(x, axis=None, keepdims=False):
    """See :func:`numpy.nanmin`."""
    return reduce_array_op(np.nanmin, x, axis=axis, keepdims=keepdims)


@implements(np.nanmax)
def nanmax(x, axis=None, keepdims=False):
    """See :func:`numpy.nanmax`."""
    return reduce_array_op(np.nanmax, x, axis=axis, keepdims=keepdims)


@implements(np.nanargmin)
def nanargmin(x, axis=None):
    """See :func:`numpy.nanargmin`."""
    return reduce_array_op(np.nanargmin, x, axis=axis)


@implements(np.nanargmax)
def nanargmax(x, axis=None):
    """See :func:`numpy.nanargmax`."""
    return reduce_array_op(np.nanargmax, x, axis=axis)


@implements(np.nansum)
def nansum(x, axis=None, dtype=None, keepdims=False):
    """See :func:`numpy.nansum`."""
    return reduce_array_op(np.nansum, x, axis=axis, dtype=dtype, keepdims=keepdims)


@implements(np.nanmean)
def nanmean(x, axis=None, dtype=None, keepdims=False):
    """See :func:`numpy.nanmean`."""
    return reduce_array_op(np.nanmean, x, axis=axis, dtype=dtype, keepdims=keepdims)


@implements(np.nanstd)
def nanstd(x, axis=None, dtype=None, ddof=0, keepdims=False):
    """See :func:`numpy.nanstd`."""
    return reduce_array_op(np.nanstd, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


@implements(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """See :func:`numpy.nan_to_num`."""
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

    Returns:
        MedicalVolume: The rounded medical image.

    Examples:
        >>> mv = MedicalVolume(10*np.random.rand(3,4,5), affine=np.eye(4))
        >>> mv_rounded = np.round(mv, decimals=3)
    """
    affine = np.around(x.affine, decimals=decimals) if affine else x.affine
    return x._partial_clone(volume=np.around(x.volume, decimals=decimals), affine=affine)


@implements(np.clip)
def clip(x, x_min, x_max, **kwargs):
    """Clip the values in the array.

    Same as applying :func:`numpy.clip` on ``x.volume``.
    Only one of ``x_min`` or ``x_max`` can be ``None``.

    Args:
        x (MedicalVolume): Medical image to clip.
        x_min (array-like or ``MedicalVolume``): Minimum value.
            If ``None``, clipping is not performed on this edge.
        x_max (array-like or ``MedicalVolume``): Maximum value.
            If ``None``, clipping is not performed on this edge.
        kwargs: Optional keyword arguments, see :func:`numpy.clip`.

    Returns:
        MedicalVolume: The clipped medical image.

    Note:
        The ``out`` positional argument is not currently supported.

    Examples:
        >>> mv = MedicalVolume([[[0,1,2,3,4,5,6,7,8,9]]], affine=np.eye(4))
        >>> np.clip(mv, 1, 5)  # Clip values between [1, 5]
        MedicalVolume(volume=[[[1 1 2 3 4 5 5 5 5 5]]])
        >>> np.clip(mv, x_max=5)  # Clip values between (-inf, 5]
        MedicalVolume(volume=[[[0 1 2 3 4 5 5 5 5 5]]])
    """
    if isinstance(x_min, MedicalVolume):
        x_min = x_min.reformat_as(x).A
    if isinstance(x_max, MedicalVolume):
        x_max = x_max.reformat_as(x).A

    arr = np.clip(x.A, x_min, x_max, **kwargs)
    return x._partial_clone(volume=arr)


@implements(np.stack)
def stack(xs, axis: int = -1):
    """Stack medical images across non-spatial dimensions.

    Images will be auto-oriented to the orientation of the first medical volume.

    Args:
        xs (array-like[MedicalVolume]): 1D array-like of aligned medical images to stack.
        axis (int, optional): Axis to stack along.

    Returns:
        MedicalVolume: The stacked medical image.

    Note:
        Unlike NumPy, the default stacking axis is ``-1``.

    Note:
        Headers are not set unless all inputs have headers of the same
        shape. This functionality may change in the future.

    Examples:
        >>> mv = dm.MedicalVolume([[[0,1,2,3]]], affine=np.eye(4))
        >>> np.stack([mv, mv], axis=-1)  # Stacks along last axis.
        MedicalVolume(volume=[[[[0 0], [1 1], [2 2], [3 3]]]])
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

    Args:
        xs (Sequence[MedicalVolume]): The medical images to concatenate.
        axis (int, optional): The axis to concatenate on.

    Returns:
        MedicalVolume: The concatenated medical image.

    Note:
        Headers are not set unless all inputs have headers of the same
        shape. This functionality may change in the future.

    Examples:
        >>> mv = dm.MedicalVolume([[[[0],[1],[2],[3]]]], affine=np.eye(4))
        >>> np.concatenate([mv, mv], axis=-1)  # Concatenate along non-spatial dimension
        MedicalVolume(volume=[[[[0 0], [1 1], [2 2], [3 3]]]])
        >>> mv2 = dm.MedicalVolume(
            [[[[4],[5],[6],[7]]]],
            affine=[[1,0,0,0], [0,1,0,0],[0,0,1,4],[0,0,0,1]]
        )
        >>> np.concatenate([mv, mv2], axis=2)  # Concatenate along spatial dimension
        MedicalVolume(volume=[[[[0]
            [1]
            [2]
            [3]
            [0]
            [1]
            [2]
            [3]]]])
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

    Examples:
        >>> mv = dm.MedicalVolume(np.random.rand(3,4,5), affine=np.eye(4))
        >>> mv_expanded = np.expand_dims(mv, axis=-1)  # Expand last dimension
        >>> mv_expanded.shape
        (3, 4, 5, 1)
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

    Examples:
        >>> mv = MedicalVolume(np.random.rand(3,4,5,1), np.eye(4))
        >>> mv_squeezed = np.squeeze(mv)  # squeeze all non-spatial dimensions
        >>> mv_squeezed.shape
        (3, 4, 5)
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


@implements(np.pad)
def pad(x: MedicalVolume, pad_width, mode="constant", **kwargs):
    """Implementation of :func:`numpy.pad` for :class:`MedicalVolume`.

    Padding a MedicalVolume can affect the affine matrix of the volume.
    When spatial dimensions are padded, the scanner origin changes.

    In addition to standard numpy syntax for ``pad_width``, this
    function provides some shortcuts for padding particular dimensions.

    Either ``None`` or ``0`` can be used to indicate a dimension should
    not be padded. For example:

    >>> mv = MedicalVolume(np.ones(3,4,5), affine=np.eye(4))
    >>> pad(mv, (None, 0, (2,3)))  # dimensions 0 and 1 will not be padded
    >>> pad(mv, ((0,0), (0,0), (2,3)))  # equivalent to previous, but in numpy syntax

    Integers can also be used to indicate the dimension should be padded by the same
    amount on both sides:

    >>> mv = MedicalVolume(np.ones(3,4,5), affine=np.eye(4))
    >>> pad(mv, (5, (1,2), (2,3)))  # dimension 0 padded by 5 on both sides
    >>> pad(mv, ((5,5), (1,2), (2,3)))  # equivalent to previous, but in numpy syntax

    ``pad_width`` can also be shorter than the total MedicalVolume dimensions.
    In this case, the padding is applying in a broadcasting fashion. For example,
    if the MedicalVolume is 3D, then specifying padding widths for only two
    dimensions will pad the last two dimensions:

    >>> mv = MedicalVolume(np.ones(3,4,5), affine=np.eye(4))
    >>> pad(mv, (4, 6))  # last dimension padded by 6, second to last padded by 4
    >>> pad(mv, ((0,0), (4,4), (6,6)))  # equivalent to previous, but in numpy syntax

    Args:
        x (MedicalVolume): The medical image.
        pad_width (Union[Sequence, int]): Same as :func:`numpy.pad`.
        mode (str): Same as :func:`numpy.pad`.
        kwargs: Same as :func:`numpy.pad`.

    Returns:
        MedicalVolume: The padded medical image.

    Note:
        Currently, headers are not preserved upon padding. The returned medical image
        will not have any headers. This may change in the future.

    Examples:
        >>> arr = np.random.rand(3,4,5)
        >>> mv = MedicalVolume(arr, affine=np.eye(4))
        >>> mv_pad = np.pad(mv, 1)  # pad all dimensions by 1
    """
    if _is_int(pad_width):
        pad_width = ((pad_width,),) * x.ndim
    if len(pad_width) < x.ndim:
        pad_width = ((0,)) * (x.ndim - len(pad_width)) + tuple(pad_width)
    pad_width = tuple((0,) if x is None else (x,) if _is_int(x) else x for x in pad_width)
    pad_width = tuple(x * 2 if len(x) == 1 else x for x in pad_width)
    assert all(len(x) == 2 for x in pad_width), pad_width

    # Update scanner origin.
    ijk = np.asarray([-p[0] for p in pad_width[:3]] + [0])
    origin = x.affine @ ijk
    affine = x.affine.copy()
    affine[:, 3] = origin

    arr = np.pad(x.A, pad_width, mode=mode, **kwargs)

    return x._partial_clone(volume=arr, affine=affine, headers=None)


@implements(np.where)
def where(*args, **kwargs):
    """See :func:`numpy.where`."""
    return np.where(np.asarray(args[0]), *args[1:], **kwargs)


@implements(np.all)
def all_np(x, axis=None, keepdims=np._NoValue):
    """See :func:`numpy.all`."""
    return reduce_array_op(np.all, x, axis=axis, keepdims=keepdims)


@implements(np.any)
def any_np(x, axis=None, keepdims=np._NoValue):
    """See :func:`numpy.any`."""
    return reduce_array_op(np.any, x, axis=axis, keepdims=keepdims)


@implements(np.zeros_like)
def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    """See :func:`numpy.zeros_like`."""
    vol = np.zeros_like(a.A, dtype=dtype, order=order, subok=subok, shape=shape)
    return a._partial_clone(volume=vol)


@implements(np.ones_like)
def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    """See :func:`numpy.ones_like`."""
    vol = np.ones_like(a.A, dtype=dtype, order=order, subok=subok, shape=shape)
    return a._partial_clone(volume=vol)


@implements(np.shares_memory)
def shares_memory(a, b, max_work=None):
    """Determine if two medical volumes share memory.

    This function implements :func:`numpy.shares_memory` for :class:`MedicalVolume`.
    Two volumes share memory if the pixel arrays and headers (if defined)
    share memory.

    Args:
        a (MedicalVolume): Input volume.
        b (MedicalVolume): Input volume.
        max_work (int, optional): Same as :func:`numpy.shares_memory`.

    Returns:
        bool: ``True`` if pixel arrays and headers (if defined) share memory.

    Raises:
        numpy.TooHardError: Exceeded max_work.

    Examples:
        >>> arr = np.random.rand(3,4,5)
        >>> mv1 = MedicalVolume(arr, affine=np.eye(4))
        >>> mv2 = MedicalVolume(arr, affine=np.eye(4))
        >>> np.shares_memory(mv1, mv2)  # Compare medicalVolume with same array in memory
        True
        >>> mv3 = MedicalVolume(arr.copy(), affine=np.eye(4))
        >>> np.shares_memory(mv1, mv2)  # Compare medicalVolume with different arrays in memory
        False
    """
    vol = np.shares_memory(a.A, b.A, max_work=max_work)
    headers = True
    if a.headers() is not None or b.headers() is not None:
        headers = np.shares_memory(a.headers(), b.headers(), max_work=max_work)
    return vol and headers


@implements(np.may_share_memory)
def may_share_memory(a, b, max_work=None):  # pragma: no cover
    """Determine if two medical volumes may share memory.

    This function implements :func:`numpy.may_share_memory` for :class:`MedicalVolume`.
    Two volumes share memory if the pixel arrays and headers (if defined)
    may share memory.

    Args:
        a (MedicalVolume): Input volume.
        b (MedicalVolume): Input volume.
        max_work (int, optional): Same as :func:`numpy.may_share_memory`.

    Returns:
        bool: ``True`` if pixel arrays and headers (if defined) may share memory.

    Raises:
        numpy.TooHardError: Exceeded max_work.

    Examples:
        >>> arr = np.random.rand(3,4,5)
        >>> mv1 = MedicalVolume(arr, affine=np.eye(4))
        >>> mv2 = MedicalVolume(arr, affine=np.eye(4))
        >>> np.shares_memory(mv1, mv2)  # Compare medicalVolume with same array in memory
        True
        >>> mv3 = MedicalVolume(arr.copy(), affine=np.eye(4))
        >>> np.shares_memory(mv1, mv2)  # Compare medicalVolume with different arrays in memory
        False
    """
    vol = np.may_share_memory(a.A, b.A, max_work=max_work)
    headers = True
    if a.headers() is not None or b.headers() is not None:
        headers = np.may_share_memory(a.headers(), b.headers(), max_work=max_work)
    return vol and headers


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


def _is_int(x):
    return isinstance(x, int) or (
        np.isscalar(x) and hasattr(x, "dtype") and np.issubdtype(x.dtype, np.integer)
    )
