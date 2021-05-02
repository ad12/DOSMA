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
    "where",
    "all_np",
    "any_np",
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


@implements(np.clip)
def clip(x, x_min, x_max, **kwargs):
    """Clip the values in the array.

    Same as applying :func:`np.clip` on ``x.volume``.
    Only one of ``x_min`` or ``x_max`` can be ``None``.

    Args:
        x (MedicalVolume): Medical image to clip.
        x_min (array-like or ``MedicalVolume``): Minimum value.
            If ``None``, clipping is not performed on this edge.
        x_max (array-like or ``MedicalVolume``): Maximum value.
            If ``None``, clipping is not performed on this edge.
        kwargs: Optional keyword arguments, see :func:`np.clip`.

    Returns:
        MedicalVolume: The clipped medical image.

    Note:
        The ``out`` positional argument is not currently supported.
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
