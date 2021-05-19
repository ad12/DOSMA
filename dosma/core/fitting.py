import inspect
import multiprocessing as mp
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize as sop
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from dosma import defaults
from dosma.core.device import cpu_device, get_array_module, get_device
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.utils import env

if env.cupy_available():
    import cupy as cp

__all__ = [
    "CurveFitter",
    "PolyFitter",
    "MonoExponentialFit",
    "curve_fit",
    "polyfit",
    "monoexponential",
    "biexponential",
]


class _Fit(ABC):
    """Abstract class for fitting quantitative values."""

    @abstractmethod
    def fit(self) -> Tuple[MedicalVolume, MedicalVolume]:
        """Fit quantitative values per pixel across multiple volumes.

        Pixels with errors in fitting are set to np.nan.

        Returns:
            tuple[MedicalVolume, MedicalVolume]: Quantitative value volume and
                r-squared goodness of fit volume.
        """
        pass  # pragma: no cover


class _Fitter(ABC):
    # This is just a summary on variables used in this abstract class,
    # the proper values/initialization should be done in child class.
    nan_to_num: Optional[float]
    out_ufuncs: Optional[Union[Callable, Sequence[Callable]]]
    out_bounds: np.ndarray  # array-like
    r2_threshold: float
    y_bounds: Optional[Tuple[float]]

    def _format_out_ufuncs(self, _out_ufuncs, _func_nparams):
        if not isinstance(_out_ufuncs, Callable) and not all(
            isinstance(ufunc, Callable) or ufunc is None for ufunc in _out_ufuncs
        ):
            raise TypeError(
                f"`out_ufuncs` must be callable or sequence of callables. Got {_out_ufuncs}"
            )

        if isinstance(_out_ufuncs, Sequence):
            if len(_out_ufuncs) > _func_nparams:
                warnings.warn(
                    f"len(out_ufuncs)={len(_out_ufuncs)}, but only {_func_nparams} parameters. "
                    f"Extra ufuncs will be ignored."
                )

        return _out_ufuncs

    def _format_out_bounds(self, _out_bounds):
        out_bounds = np.asarray(_out_bounds)
        if out_bounds.shape[-1] != 2 or out_bounds.ndim > 2:
            raise ValueError("Invalid `out_bounds` - shape must be ([num_params,] 2)")
        if np.any(out_bounds[..., 0] > out_bounds[..., 1]):
            raise ValueError("Invalid `out_bounds` - lower bound must be <= upper bound")
        return out_bounds

    def _format_r2_threshold(self, _r2_threshold):
        if isinstance(_r2_threshold, str):
            if _r2_threshold != "preferences":
                raise ValueError(
                    f"Invalid value r2_threshold='{_r2_threshold}'. "
                    f"Expected `None`, a number between [0, 1], or 'preferences'."
                )
            _r2_threshold = preferences.fitting_r2_threshold
        return _r2_threshold

    def _process_mask(self, mask, y: MedicalVolume):
        """Process mask into appropriate shape."""
        arr_types = (np.ndarray, cp.ndarray) if env.cupy_available() else (np.ndarray,)
        if isinstance(mask, arr_types):
            mask = y._partial_clone(volume=mask, headers=None)
        elif not isinstance(mask, MedicalVolume):
            raise TypeError("`mask` must be a MedicalVolume or ndarray")

        mask = mask.reformat_as(y)
        if not mask.is_same_dimensions(y, defaults.AFFINE_DECIMAL_PRECISION):
            raise RuntimeError("`mask` and `y` dimension mismatch")

        return mask > 0

    def _process_params(self, x, r_squared):
        """
        Applies ``self.out_ufuncs`` and ``self.out_bounds``, in that order.
        Input is modified in place.

        Returns:
            ndarray: Values outside of ``self.out_bounds`` will be set to np.nan.
        """
        nan_to_num = self.nan_to_num
        out_ufuncs = self.out_ufuncs
        out_bounds = self.out_bounds
        r2_threshold = self.r2_threshold
        nparams = x.shape[-1]

        if isinstance(out_ufuncs, Callable):
            x = out_ufuncs(x)
        elif isinstance(out_ufuncs, Sequence):
            for i in range(min(nparams, len(out_ufuncs))):
                if out_ufuncs[i] is not None:
                    x[..., i] = out_ufuncs[i](x[..., i])

        if out_bounds is not None:
            if out_bounds.ndim == 2:
                extra_bounds = [(-np.inf, np.inf) for _ in range(nparams - out_bounds.shape[0])]
                if len(extra_bounds) > 0:
                    extra_bounds = np.stack(extra_bounds, axis=0)
                    out_bounds = np.concatenate([out_bounds, extra_bounds], axis=0)
                out_bounds = out_bounds.T
            lb, ub = out_bounds[0], out_bounds[1]
            x[(x < lb) | (x > ub)] = np.nan

        if r2_threshold is not None:
            x[(r_squared < r2_threshold)] = np.nan

        if nan_to_num is not None:
            x = np.nan_to_num(x, nan=nan_to_num, copy=False)

        return x

    def _fit(self, x, y):
        """Internal fitting function.

        Returns:
            tuple[array-like, array-like]: The fitted parameters and corresponding
                goodness of fit (:math:`r^2`).
        """
        raise NotImplementedError  # pragma: no cover

    def fit(self, x, y: Sequence[MedicalVolume], mask=None, copy_headers: bool = True, **kwargs):
        """
        Args:
            x (array-like): 1D array of independent variables corresponding to different ``y``.
            y (list[MedicalVolumes]): Dependent variable (in order) corresponding to values of
                independent variable ``x``. Data must be spatially aligned.
            mask (``MedicalVolume`` or ``ndarray``, optional): If specified, only entries where
                ``mask > 0`` will be fit.
            copy_headers (bool): If ``True``, headers will be deep copied. If ``False``,
                headers will not be copied. Returned values will not have headers.
            kwargs: Optional keyword arguments to be passed to :func:`self._fit`.

        Returns:
            Tuple[MedicalVolume, MedicalVolume]: Tuple of fitted parameters (``popt``)
            and goodness of fit (``r2``) values. Last axis of fitted parameters
            corresponds to different parameters in order of appearance in ``self.func``.
        """
        svs = []

        if (not isinstance(y, (list, tuple))) or (
            not all([isinstance(_y, MedicalVolume) for _y in y])
        ):
            raise TypeError("`y` must be sequence of MedicalVolumes.")

        x = np.asarray(x)
        if x.shape[-1] != len(y):
            raise ValueError(
                "Dimension mismatch: x.shape[-1]={:d}, but len(y)={:d}".format(x.shape[-1], len(y))
            )

        orientation = y[0].orientation
        y = [_y.reformat(orientation) for _y in y]

        if mask is not None:
            mask = self._process_mask(mask, y[0])
            mask = mask.volume.reshape(-1)

        original_shape = y[0].shape
        svs = np.concatenate([_y.volume.reshape((1, -1)) for _y in y], axis=0)
        flattened_shape = svs.shape

        # Select indices to fit.
        if mask is not None:
            svs = svs[:, mask]

        popt, r_squared = self._fit(x, svs, **kwargs)
        popt = self._process_params(popt, r_squared)

        if mask is not None:
            popt_full = np.empty(flattened_shape[-1:] + popt.shape[-1:])
            r2_full = np.empty(flattened_shape[-1])
            nan_val = np.nan if self.nan_to_num is None else self.nan_to_num
            popt_full.fill(nan_val)
            r2_full.fill(nan_val)
            popt_full[mask] = popt
            r2_full[mask] = r_squared
            popt = popt_full
            r_squared = r2_full
            del popt_full, r2_full  # variables not used later - drop reference to underlying array

        popt = popt.reshape(original_shape + popt.shape[-1:])
        r_squared = r_squared.reshape(original_shape)

        # For parameters, headers have to be deep copied and expanded to follow
        # broadcasting dimensions.
        if copy_headers:
            headers = y[0].headers()
            if headers is not None:
                headers = deepcopy(headers)
                if popt.ndim > y[0].volume.ndim:
                    axis = tuple(-i for i in range(1, popt.ndim - y[0].volume.ndim + 1))
                    headers = np.expand_dims(headers, axis=axis)
            popt_headers, r2_headers = headers, True
        else:
            popt_headers, r2_headers = None, None
        popt = y[0]._partial_clone(volume=popt, headers=popt_headers)
        rsquared_volume = y[0]._partial_clone(volume=r_squared, headers=r2_headers)

        return popt, rsquared_volume


class CurveFitter(_Fitter):
    """The class using non-linear least squares to fit a function to data.

    This class is a wrapper around the :func:`dosma.utils.fits.curve_fit` function
    that handles :class:`MedicalVolume` data and supports additional post-processing
    on fitted parameters.

    ``self.fit(x, y, mask=None)`` will fit independent variables ``x`` with observed
    MedicalVolumes ``y``. To only fit certain regions of volumes ``y``, specify ``mask``.

    Args:
        func (callable): The model function, f(x, ...). It must take the independent variable
            as the first argument and the parameters to fit as separate remaining arguments.
        p0 (Sequence, optional): Initial guess for the parameters (length N). If None, then
            the initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a ValueError is raised).
        y_bounds (tuple, optional): Lower and upper bound on y values. Defaults to no bounds.
            Sequences with observations out of this range will not be processed.
        out_ufuncs (Callable, Sequence[Callable]): Function(s)
            to post-process parameter maps. Defaults to no post-processing.
            Each function must operate on ndarrays in an element-by-element fashion,
            should only take one argument, which will be the parameter map, and output
            the processed parameter map (an ndarray). If ``isinstance(out_ufuncs, Callable)``,
            operates on map of all parameters, where last dimension (``axis=-1``) corresponds to
            different parameters. If ``isinstance(out_ufuncs, Sequence)``, each ufunc corresponds
            to one parameter.
        out_bounds (array-like, optional): Lower and upper bounds (inclusive) on fitted parameters.
            Defaults to no bounds. Fitted parameters outside of this range will be set to
            ``np.nan``. Last dimension should have size 2. If ``out_bounds.ndim == 1``, bounds
            will be applied to all parameters.
        r2_threshold (float): Minimum :math:`r^2` goodness of fit value to accept fit per sample.
            Parameter values below this threshold will be set to np.nan.
            Defaults to ``preferences.fitting_r2_threshold``. To ignore, set to ``None``.
        nan_to_num (float, optional): If specified, all fitted parameters equal to ``np.nan``
            will be converted to this value. This uses ``numpy.nan_to_num``, which will also
            replace *posinf* and *neginf* values with very large positive and negative values,
            respectively. See ``numpy.nan_to_num`` for more details.
        num_workers (int, optional): Maximum number of workers to use for fitting.
        chunksize (int, optional): Size of chunks sent to worker processes when
            ``num_workers > 0``. When ``show_pbar=True``, this defaults to the standard
            value in :func:`tqdm.concurrent.process_map`.
        verbose (bool, optional): If `True`, show progress bar. Note this can increase runtime
            slightly when using multiple workers.
        kwargs: Keyword args for :func:`dosma.utils.fits.curve_fit`.

    Examples:
        Fitting :func:`monoexponential` (:math:`y = a * e^{b*x}`) to independent variables ``x``
        and dependent variables ``y`` with initial guesses ``a=1.0`` and ``b=-0.2``:

        >>> fitter = CurveFitter(monoexponential, p0=(1.0, -0.2))
        >>> popt, r2 = fitter.fit(x, y)
        >>> a_hat, b_hat = popt[..., 0], popt[..., 1]

        Post-process ``b`` by taking the inverse of its absolute value of ``b``
        (i.e. :math:`\\frac{1}{|b|}`). Set all values not in domain
        :math:`0 \\leq \\frac{1}{|b|} \\leq 100` or with a goodness of fit less than 0.9
        (:math:`r^2` < 0.9) to ``np.nan``:

        >>> ufunc = lambda x: 1 / np.abs(x)
        >>> out_bounds = ((-np.inf, np.inf), (0, 100))
        >>> fitter = CurveFitter(
        ...     monoexponential, p0=(1.0, -0.2), out_ufuncs=[None, ufunc], out_bounds=out_bounds)
        >>> popt, r2 = fitter.fit(x, y)
        >>> a_hat, inv_abs_b_hat = popt[..., 0], popt[..., 1]
    """

    def __init__(
        self,
        func: Callable,
        p0: Sequence[float] = None,
        y_bounds: Tuple[float] = None,
        out_ufuncs: Union[Callable, Sequence[Callable]] = None,
        out_bounds=None,
        r2_threshold: float = "preferences",
        nan_to_num: float = None,
        num_workers: int = 0,
        chunksize: int = None,
        verbose: bool = False,
        **kwargs,
    ):
        func_name = func.__name__ if hasattr(func, "__name__") else type(func).__name__
        sig = inspect.signature(func)
        func_args = list(sig.parameters)
        func_nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1

        if out_ufuncs is not None:
            out_ufuncs = self._format_out_ufuncs(out_ufuncs, func_nparams)

        if out_bounds is not None:
            out_bounds = self._format_out_bounds(out_bounds)

        r2_threshold = self._format_r2_threshold(r2_threshold)

        self._func = func
        self._func_name = func_name
        self.p0 = self._format_p0(p0)
        self.y_bounds = y_bounds
        self.out_ufuncs = out_ufuncs
        self.out_bounds = out_bounds
        self.r2_threshold = r2_threshold
        self.nan_to_num = nan_to_num
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.verbose = verbose
        self.kwargs = kwargs

    def _format_p0(
        self, p0, ref: MedicalVolume = None, flatten: bool = False, mask=None, depth: int = 0
    ):
        if p0 is None or isinstance(p0, Number):
            return p0
        elif isinstance(p0, MedicalVolume) and depth > 0:
            if ref is not None:
                p0 = p0.reformat_as(ref)
                assert p0.is_same_dimensions(ref, err=True)
            if flatten:
                p0 = p0.A.flatten()
                if mask is not None:
                    p0 = p0[mask]
            return p0
        elif isinstance(p0, np.ndarray) and depth > 0:
            if ref is not None and p0.shape != ref.shape:
                raise ValueError(f"Got p0.shape={p0.shape}, but y.shape={ref.shape}")
            if flatten:
                p0 = p0.flatten()
            if mask is not None:
                p0 = p0[mask]
            return p0

        if isinstance(p0, Mapping):
            return {k: self._format_p0(v, ref, flatten, mask, depth + 1) for k, v in p0.items()}
        elif isinstance(p0, Sequence):
            return tuple(self._format_p0(v, ref, flatten, mask, depth + 1) for v in p0)
        elif isinstance(p0, (np.ndarray, MedicalVolume)):
            # If a single numpy array is provided as the parameter input, the
            # last axis is considered to be the parameter axis. Note, this may
            # change in the future.
            return tuple(
                self._format_p0(p0[..., i], ref, flatten, mask, depth + 1)
                for i in range(p0.shape[-1])
            )

        raise ValueError(f"p0={p0} not supported")

    def fit(
        self, x, y: Sequence[MedicalVolume], mask=None, p0=np._NoValue, copy_headers: bool = True
    ):
        """Perform non-linear least squares fit.

        Args:
            x (array-like): 1D array of independent variables corresponding to different ``y``.
            y (list[MedicalVolumes]): Dependent variable (in order) corresponding to values of
                independent variable ``x``. Data must be spatially aligned.
            mask (``MedicalVolume`` or ``ndarray``, optional): If specified, only entries where
                ``mask > 0`` will be fit.
            p0 (Sequence, optional): Initial guess for the parameters.
                Defaults to ``self.p0``.
            copy_headers (bool, optional): If ``True``, headers will be deep copied. If ``False``,
                headers will not be copied. Returned values will not have headers.

        Returns:
            Tuple[MedicalVolume, MedicalVolume]: Tuple of fitted parameters (``popt``)
            and goodness of fit (``r2``) values. Last axis of fitted parameters
            corresponds to different parameters in order of appearance in ``self.func``.
        """
        if get_device(x) != cpu_device:
            raise RuntimeError("`x` must be on the CPU")
        if any(get_device(_y) != cpu_device for _y in y):
            raise RuntimeError("All elements in `y` must be on the CPU")

        if mask is not None:
            mask = self._process_mask(mask, y[0])

        if p0 is np._NoValue:
            p0 = self.p0
        p0 = self._format_p0(
            p0,
            ref=y[0],
            flatten=True,
            mask=mask.A.reshape(-1) if mask is not None else None,
        )

        return super().fit(x, y, mask=mask, p0=p0, copy_headers=copy_headers)

    def _fit(self, x, y, p0=np._NoValue):
        assert p0 is not np._NoValue

        return curve_fit(
            self._func,
            x,
            y,
            self.y_bounds,
            p0=p0,
            show_pbar=self.verbose,
            num_workers=self.num_workers,
            chunksize=self.chunksize,
            **self.kwargs,
        )

    def __str__(self) -> str:
        chunk_size = 1
        attrs = [
            "p0",
            "y_bounds",
            "out_bounds",
            "r2_threshold",
            "nan_to_num",
            "num_workers",
            "chunksize",
            "verbose",
        ]
        vals = [f"func={self._func_name}"]
        vals += [f"{k}={getattr(self, k)}" for k in attrs]
        vals += [f"{k}={v}" for k, v in self.kwargs.items()]
        val_str = "\n\t".join(
            [
                ", ".join(vals[i * chunk_size : (i + 1) * chunk_size]) + ","
                for i in range(int(np.ceil(len(vals) / chunk_size)))
            ]
        )
        return f"{self.__class__.__name__}(\n\t" + val_str + "\n)"


class PolyFitter(_Fitter):
    """The class using linear least squares to fit a polynomial to data.

    This class is a wrapper around the :func:`dosma.utils.fits.polyfit` function
    that handles :class:`MedicalVolume` data and supports additional post-processing
    on fitted parameters. Most attributes are shared with :class:`dosma.utils.fits.CurveFitter`.
    See that class for details.

    Unlike non-linear curve fitting, polynomial fitting can be done for multiple data sequences
    (i.e. multiple pixels/voxels) as a single least squares problem, which is the default in
    :mod:`numpy`. However, poor matrix conditioning can result in poor parameter estimates.
    As a result, this class also supports fitting each data sequence separately.
    If ``num_workers = None``, data sequences will be fit as a single least squares problem.
    If ``num_workers = 0``, data sequences will be fit separately and sequentially.
    If ``num_workers > 0``, multiprocessing will be used to fit each data sequence separately
    and in parallel.

    Args:
        deg (int): Degree of the fitting polynomial. See :func:`numpy.polyfit`.
        rcond (float, optional): Relative condition number of the fit. See :func:`numpy.polyfit`.
        y_bounds (tuple, optional): Same as :class:`CurveFitter`.
        out_ufuncs (Callable, Sequence[Callable]): Same as :class:`CurveFitter`.
        out_bounds (array-like, optional): Same as :class:`CurveFitter`.
        r2_threshold (float): Same as :class:`CurveFitter`.
        nan_to_num (float, optional): Same as :class:`CurveFitter`.
        num_workers (int, optional): Maximum number of workers to use for fitting.
            If ``None``, all data sequences should be solved as a single least squares problem.
            If ``0``, each data sequence will be fit separately from one another.
            Defaults to ``None``.
        chunksize (int, optional): Same as :class:`CurveFitter`.
        verbose (bool, optional): Same as :class:`CurveFitter`.

    Examples:
        Fitting polynomial of degree 1 (:math:`y = a*x + b`) to to independent variables ``x``
        and dependent variables ``y``.

        >>> fitter = PolyFitter(deg=1)
        >>> popt, r2 = fitter.fit(x, y)
        >>> a_hat, b_hat = popt[..., 0], popt[..., 1]

        Post-process ``b`` by taking the absolute value of ``b`` (i.e. :math:`|b|`).
        Set all values not in domain :math:`0 \\leq |b| \\leq 100` or with a goodness of fit
        less than 0.9 (:math:`r^2` < 0.9) to ``np.nan``:

        >>> ufunc = lambda x: np.abs(x)
        >>> out_bounds = ((-np.inf, np.inf), (0, 100))
        >>> fitter = CurveFitter(
        ...     monoexponential, p0=(1.0, -0.2), out_ufuncs=[None, ufunc], out_bounds=out_bounds)
        >>> popt, r2 = fitter.fit(x, y)
        >>> a_hat, abs_b_hat = popt[..., 0], popt[..., 1]
    """

    def __init__(
        self,
        deg: int,
        rcond: float = None,
        y_bounds: Tuple[float] = None,
        out_ufuncs: Union[Callable, Sequence[Callable]] = None,
        out_bounds=None,
        r2_threshold: float = "preferences",
        nan_to_num: float = None,
        num_workers: int = None,
        chunksize: int = None,
        verbose: bool = False,
    ):
        if out_ufuncs is not None:
            out_ufuncs = self._format_out_ufuncs(out_ufuncs, deg + 1)

        if out_bounds is not None:
            out_bounds = self._format_out_bounds(out_bounds)

        r2_threshold = self._format_r2_threshold(r2_threshold)

        self.deg = deg
        self.rcond = rcond

        self.y_bounds = y_bounds
        self.out_ufuncs = out_ufuncs
        self.out_bounds = out_bounds
        self.r2_threshold = r2_threshold
        self.nan_to_num = nan_to_num
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.verbose = verbose

    def fit(self, x, y: Sequence[MedicalVolume], mask=None, copy_headers: bool = True):
        """Perform linear least squares fit.

        Args:
            x (array-like): Same as :meth:`CurveFitter.fit`.
            y (Sequence[MedicalVolume]): Same as :meth:`CurveFitter.fit`.
            mask (MedicalVolume or ndarray): Same as :meth:`CurveFitter.fit`.
            copy_headers (bool, optional): If ``True``, headers will be deep copied. If ``False``,
                headers will not be copied. Returned values will not have headers.

        Returns:
            Tuple[MedicalVolume, MedicalVolume]: Tuple of fitted parameters (``popt``)
                and goodness of fit (``r2``) values (same as `CurveFitter.fit`). The last
                axis of ``popt`` corresponds to different polynomial parameters in order
                ``y = popt[..., 0] * x ** self.deg + ... + popt[..., self.deg-1]``.
        """
        device = get_device(x)
        y_devices = [get_device(_y) for _y in y]
        if any(_y_device != device for _y_device in y_devices):
            raise RuntimeError(
                f"All elements in `y` must be on the same device as `x` ({device}). "
                f"Got {y_devices}."
            )

        return super().fit(x, y, mask=mask, copy_headers=copy_headers)

    def _fit(self, x, y):
        return polyfit(
            x,
            y,
            deg=self.deg,
            rcond=self.rcond,
            y_bounds=self.y_bounds,
            show_pbar=self.verbose,
            num_workers=self.num_workers,
            chunksize=self.chunksize,
        )

    def __str__(self) -> str:
        chunk_size = 1
        attrs = [
            "deg",
            "rcond",
            "y_bounds",
            "out_bounds",
            "r2_threshold",
            "nan_to_num",
            "num_workers",
            "chunksize",
            "verbose",
        ]
        vals = [f"{k}={getattr(self, k)}" for k in attrs]
        val_str = "\n\t".join(
            [
                ", ".join(vals[i * chunk_size : (i + 1) * chunk_size]) + ","
                for i in range(int(np.ceil(len(vals) / chunk_size)))
            ]
        )
        return f"{self.__class__.__name__}(\n\t" + val_str + "\n)"


class MonoExponentialFit(_Fit):
    """Fit quantitative values using mono-exponential fit of model :math:`y=a*exp(-x/tc)`.

    Args:
        x (:obj:`array-like`): 1D array of independent variables corresponding to different volumes.
        y (list[MedicalVolumes]): Volumes (in order) corresponding to independent variable in `x`.
        mask (:obj:`MedicalVolume`, optional): Mask of pixels to fit.
            If specified, pixels outside of mask region are ignored and set to ``np.nan``.
            Speeds fitting time as fewer fits are required.
        bounds (:obj:`tuple[float, float]`, optional): Upper and lower bound for quantitative
            values. Values outside those bounds will be set to ``np.nan``.
        tc0 (:obj:`float`, optional): Initial time constant guess. If ``"polyfit"``, this
            guess will be determined by first doing a polynomial fit on the log-linearized
            form of the monoexponential equation :math:`\\log y = \\log a - \\frac{t}{tc}`.
            Note for polyfit initialization, ``subvolumes`` should not have any nan or infinite
            values.
        decimal_precision (:obj:`int`, optional): Rounding precision after the decimal point.
        num_workers (int, optional): Maximum number of workers to use for fitting.
        chunksize (int, optional): Size of chunks sent to worker processes when
            ``num_workers > 0``. When ``show_pbar=True``, this defaults to the standard
            value in :func:`tqdm.concurrent.process_map`.
        verbose (bool, optional): If `True`, show progress bar. Note this can increase runtime
            slightly when using multiple workers.
    """

    def __init__(
        self,
        x: Sequence[float] = None,
        y: Sequence[MedicalVolume] = None,
        mask: MedicalVolume = None,
        bounds: Tuple[float] = (0, 100.0),
        tc0: Union[float, str] = 30.0,
        r2_threshold: float = "preferences",
        decimal_precision: int = 1,
        num_workers: int = 0,
        chunksize: int = 1000,
        verbose: bool = False,
    ):

        self.x = x
        if y is not None:
            warnings.warn(
                f"Setting `y` in the constructor can result in significant memory overhead. "
                f"Specify `y` in `{type(self).__name__}.fit(y=...)` instead."
            )
            self._check_y(x, y)
        self.y = y

        if mask is not None:
            warnings.warn(
                f"Setting `mask` in the constructor can result in significant memory overhead. "
                f"Specify `mask` in `{type(self).__name__}.fit(mask=...)` instead."
            )
        self.mask = mask

        if not (isinstance(tc0, Number) or (isinstance(tc0, str) and tc0 == "polyfit")):
            raise ValueError("`tc0` must either be a float or the string 'polyfit'.")

        self.verbose = verbose
        self.num_workers = num_workers

        if len(bounds) != 2:
            raise ValueError("`bounds` should provide lower/upper bound in format (lb, ub)")
        self.bounds = bounds

        self.chunksize = chunksize
        self.r2_threshold = r2_threshold
        self.tc0 = tc0
        self.decimal_precision = decimal_precision
        self._eps = 1e-10  # epsilon for polyfit - do not change this

    def fit(self, x=None, y: Sequence[MedicalVolume] = None, mask=None):
        """Perform monoexponential fitting.

        Returns:
            Tuple[MedicalVolume, MedicalVolume]:

                time_constant_volume (MedicalVolume): The per-voxel tc fit.
                rsquared_volume (MedicalVolume): The per-voxel r2 goodness of fit.
        """
        x = self.x if x is None else x
        y = self.y if y is None else y
        mask = self.mask if mask is None else mask

        self._check_y(x, y)
        orientation = y[0].orientation
        y = [sv.reformat(orientation) for sv in y]

        if isinstance(mask, np.ndarray):
            mask = MedicalVolume(mask, affine=y[0].affine)
            if not isinstance(mask, MedicalVolume):
                raise TypeError("`mask` must be a MedicalVolume")
        mask = mask.reformat(orientation) if mask else None

        if self.tc0 == "polyfit":
            polyfitter = PolyFitter(
                1,
                r2_threshold=0,
                num_workers=None,
                nan_to_num=0.0,
                chunksize=self.chunksize,
                verbose=self.verbose,
            )
            vols = [
                sv.astype(np.float32) if np.issubdtype(sv.dtype, np.integer) else sv for sv in y
            ]
            vols = [sv + self._eps * (sv == 0) for sv in vols]
            assert all(np.all(v != 0) for v in vols)
            vols = [np.log(v) for v in vols]
            params, _ = polyfitter.fit(x, vols, mask=mask, copy_headers=False)
            p0 = {"a": np.exp(params[..., 1]), "b": params[..., 0]}
            del vols  # begin garbage collection for large arrays sooner
        else:
            p0 = {"a": 1.0, "b": -1 / self.tc0}

        curve_fitter = CurveFitter(
            monoexponential,
            y_bounds=None,
            out_ufuncs=(None, lambda _x: 1 / np.abs(_x)),
            out_bounds=((-np.inf, np.inf), self.bounds),
            r2_threshold=self.r2_threshold,
            num_workers=self.num_workers,
            chunksize=self.chunksize,
            verbose=self.verbose,
            nan_to_num=0.0,
        )
        popt, r_squared = curve_fitter.fit(x, y, mask=mask, p0=p0)
        tc_map = popt[..., 1]

        if self.decimal_precision is not None:
            tc_map = np.around(tc_map, self.decimal_precision)

        return tc_map, r_squared

    def _check_y(self, x, y):
        if (not isinstance(y, Sequence)) or (not all([isinstance(sv, MedicalVolume) for sv in y])):
            raise TypeError("`y` must be list of MedicalVolumes.")

        if any(x.device != cpu_device for x in y):
            raise RuntimeError("All MedicalVolumes must be on the CPU")

        if len(x) != len(y):
            raise ValueError("`len(x)`={:d}, but `len(y)`={:d}".format(len(x), len(y)))


__EPSILON__ = 1e-8


def curve_fit(
    func,
    x,
    y,
    y_bounds=None,
    p0=None,
    maxfev=100,
    ftol=1e-5,
    eps=1e-8,
    show_pbar=False,
    num_workers=0,
    chunksize: int = None,
    **kwargs,
):
    """Use non-linear least squares to fit a function ``func`` to data.

    Uses :func:`scipy.optimize.curve_fit` backbone.

    Args:
        func (callable): The model function, f(x, ...). It must take the independent variable
            as the first argument and the parameters to fit as separate remaining arguments.
        x (ndarray): The independent variable(s) where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for functions
            with k predictors, but can actually be any object.
        y (ndarray): The dependent data, a length M array - nominally func(xdata, ...) - or
            an (M,N)-shaped array for N different sequences.
        y_bounds (tuple, optional): Lower and upper bound on y values. Defaults to no bounds.
            Sequences with observations out of this range will not be processed.
        p0 (Number | Sequence[Number] | ndarray | Dict, optional): Initial guess for the parameters.
            If sequence (e.g. list, tuple, 1d ndarray), it should have length P, which is the
            number of parameters. If this is a 2D numpy array, it should have a shape ``(N, P)``.
            If ``None``, then the initial values will all be 1.
        maxfev (int, optional): Maximum number of function evaluations before the termination.
            If `bounds` argument for `scipy.optimize.curve_fit` is specified, this corresponds
            to the `max_nfev` in the least squares algorithm
        ftol (float): Tolerance for termination by the change of the cost function.
            See `scipy.optimize.least_squares` for more details.
        eps (float, optional): Epsilon for computing r-squared.
        show_pbar (bool, optional): If `True`, show progress bar. Note this can increase runtime
            slightly when using multiple workers.
        num_workers (int, optional): Maximum number of workers to use for fitting.
        chunksize (int, optional): Size of chunks sent to worker processes when
            ``num_workers > 0``. When ``show_pbar=True``, this defaults to the standard
            value in :func:`tqdm.concurrent.process_map`.
        kwargs: Keyword args for `scipy.optimize.curve_fit`.

    Returns:
        Tuple[ndarray, ndarray]:

            popts (ndarray): A NxP matrix of fitted values. The last dimension (``axis=-1``)
            corresponds to the different parameters (in order).

            rsquared (ndarray): A (N,) length matrix of r-squared goodness-of-fit values.
    """
    if (get_device(x) != cpu_device) or (get_device(y) != cpu_device):
        raise RuntimeError("`x` and `y` must be on CPU")

    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(y.shape + (1,))
    N = y.shape[-1]

    func_args = list(inspect.signature(func).parameters)
    nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1
    param_args = func_args[2:] if "self" in func_args else func_args[1:]

    if p0 is None:
        p0_scalars, p0_seq = None, None
    else:
        p0_scalars, p0_seq = _format_p0(p0, param_args, N)

    if "bounds" not in kwargs:
        kwargs["maxfev"] = maxfev
    elif "max_nfev" not in kwargs:
        kwargs["max_nfev"] = maxfev

    num_workers = min(num_workers, N)
    fitter = partial(
        _curve_fit,
        x=x,
        func=func,
        y_bounds=y_bounds,
        p0=p0_scalars,
        ftol=ftol,
        eps=eps,
        nparams=nparams,
        **kwargs,
    )

    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob:
        warnings.warn("Out of bounds values found. Failure in fit will result in np.nan")

    y_T = y.T
    if p0_seq:
        y_T = [{"y": y_T[i], "p0": {k: v[i] for k, v in p0_seq.items()}} for i in range(N)]

    popts = []
    r_squared = []
    if not num_workers:
        for i in tqdm(range(N), disable=not show_pbar):
            popt_, r2_ = fitter(y_T[i])
            popts.append(popt_)
            r_squared.append(r2_)
    else:
        if show_pbar:
            tqdm_kwargs = {"chunksize": chunksize}
            tqdm_kwargs = {k: v for k, v in tqdm_kwargs.items() if v is not None}
            data = process_map(fitter, y_T, max_workers=num_workers, **tqdm_kwargs)
        else:
            with mp.Pool(num_workers) as p:
                data = p.map(fitter, y_T, chunksize=chunksize)
        popts, r_squared = [x[0] for x in data], [x[1] for x in data]

    return np.stack(popts, axis=0), np.asarray(r_squared)


def polyfit(
    x,
    y,
    deg: int,
    rcond=None,
    full=False,
    w=None,
    cov=False,
    eps=1e-8,
    y_bounds=None,
    show_pbar=False,
    num_workers=None,
    chunksize: int = None,
):
    """Use linear least squares to fit a polynomial of degree ``deg`` to data.

    This function is a wrapper around the :func:`numpy.polyfit` and :func:`cupy.polyfit`
    functions.

    In addition to standard polyfit functionality, this function also supports
    multiprocessing of the data using multiple workers. When multiprocessing is enabled,
    each data sequence is fit using a separate worker.

    In most cases, multiprocessing is not needed for linear least squares as most
    computations can be done quickly via matrix decomposition. However, there are cases
    where ``y`` values can cause low condition numbers and/or invalid outputs. In these cases,
    solving for each data sequence separately can be beneficial.

    Solving each data sequence separately without multiprocessing can also be done by setting
    ``num_workers=0``. This can be useful when certain data sequences are ill conditioned.

    Args:
        x (ndarray): The independent variable(s) where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for functions
            with k predictors, but can actually be any object.
        y (ndarray): The dependent data, a length M array - nominally func(xdata, ...) - or
            an (M,N)-shaped array for N different sequences.
        deg (int): Degree of the fitting polynomial. Same as :func:`numpy.polyfit`.
        rcond (float, optional): Same as :func:`numpy.polyfit`.
        full (bool, optional): Same as :func:`numpy.polyfit`.
        w (array-like, optional): Same as :func:`numpy.polyfit`.
        cov (bool, optional): Same as :func:`numpy.polyfit`.
        eps (float, optional): Epsilon for computing r-squared.
        y_bounds (tuple, optional): Same as :func:`curve_fit`.
        show_pbar (bool, optional): Same as :func:`curve_fit`.
        num_workers (int, optional): Maximum number of workers to use for fitting.
            If ``None``, all data sequences should be solved as a single least squares problem.
            If ``0``, each data sequence will be fit separately from one another.
            Defaults to ``None``.
        chunksize (int, optional): Same as :func:`curve_fit`.
            Only used when ``num_workers`` is not ``None``.
    """

    def _compute_r2_matrix(_x, _y, _popts):
        """
        Here, ``M`` refers to # sample points, ``K`` refers to # sequences
        This function needs to be run under the correct context
        Args:
            _x: array_like, shape (M,)
            _y: array_like, shape (M, K)
            _popts (array-like): Shape (deg+1, K)
        """
        xp = get_array_module(_y)

        _x = _x.flatten()
        _xs = xp.stack([_x ** i for i in range(len(_popts) - 1, -1, -1)], axis=-1)
        yhat = _xs @ _popts  # (M, K)

        residuals = yhat - _y
        ss_res = xp.sum(residuals ** 2, axis=0)
        ss_tot = xp.sum((_y - xp.mean(_y, axis=0, keepdims=True)) ** 2, axis=0)
        return 1 - (ss_res / (ss_tot + eps))

    x_device, y_device = get_device(x), get_device(y)
    if x_device != y_device:
        raise ValueError(f"`x` ({x_device}) and `y` ({x_device}) must be on the same device")

    scatter_data = num_workers is not None
    if (cov or full) and scatter_data:
        raise ValueError("`cov` or `full` cannot be used with multiprocessing")

    xp = get_array_module(x)

    x = xp.asarray(x)
    y = xp.asarray(y)
    if y.ndim == 1:
        y = y.reshape(y.shape + (1,))
    N = y.shape[-1]

    num_workers = min(num_workers, N) if num_workers is not None else None

    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob:
        warnings.warn("Out of bounds values found. Failure in fit will result in np.nan")

    fitter = partial(
        _polyfit, x=x, deg=deg, y_bounds=y_bounds, rcond=rcond, w=w, eps=eps, xp=xp.__name__
    )

    residuals, rank, singular_values, rcond, V = None, None, None, None, None
    with x_device:
        if not scatter_data:
            # Fit all sequences as one least squares problem.
            out = xp.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)
            if full:
                popts, residuals, rank, singular_values, rcond = out
            elif cov:
                popts, V = out
            else:
                popts = out
            r_squared = _compute_r2_matrix(x, y, popts)
            popts = popts.T
        elif num_workers == 0:
            # Fit each sequence separately
            popts = []
            r_squared = []
            for i in tqdm(range(N), disable=not show_pbar):
                popt_, r2_ = fitter(y[:, i])
                popts.append(popt_)
                r_squared.append(r2_)
            popts = xp.stack(popts, axis=0)
        else:
            # Multiprocessing - fits each sequence separately
            if show_pbar:
                tqdm_kwargs = {"chunksize": chunksize}
                tqdm_kwargs = {k: v for k, v in tqdm_kwargs.items() if v is not None}
                data = process_map(fitter, y.T, max_workers=num_workers, **tqdm_kwargs)
            else:
                with mp.Pool(num_workers) as p:
                    data = p.map(fitter, y.T, chunksize=chunksize)
            popts, r_squared = [x[0] for x in data], [x[1] for x in data]
            popts = xp.stack(popts, axis=0)

        r_squared = xp.asarray(r_squared)

    if full:
        return popts, r_squared, residuals, rank, singular_values, rcond
    elif cov:
        return popts, r_squared, V
    else:
        return popts, r_squared


def monoexponential(x, a, b):
    """Function: :math:`f(x) = a * e^{b*x}`."""
    return a * np.exp(b * x)


def biexponential(x, a1, b1, a2, b2):
    """Function: :math:`f(x) = a1*e^{b1*x} + a2*e^{b2*x}`."""
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x)


def _curve_fit(
    y_or_dict, x, func, y_bounds=None, p0=None, ftol=1e-5, eps=1e-8, nparams=None, **kwargs
):
    def _fit_internal(_x, _y):
        popt, _ = sop.curve_fit(func, _x, _y, p0=p0, ftol=ftol, **kwargs)

        residuals = _y - func(_x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((_y - np.mean(_y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + eps))

        return popt, r_squared

    def _parse_p0(_dict):
        if "p0" not in _dict or _dict["p0"] in [None, {}]:
            return p0
        _p0_dict = _dict["p0"]
        if p0 is None:
            return tuple(_p0_dict.values())
        elif isinstance(p0, Dict):
            # In Python>=3.6 dictionary key/value pairs are ordered by default.
            # The order of the keys should reflect the order of the parameters
            # in the target function.
            p0_copy = p0.copy()
            p0_copy.update(_p0_dict)
            return tuple(p0_copy.values())
        else:
            assert False  # p0 must be None or a mapping

    if nparams is None:
        func_args = inspect.getargspec(func).args
        nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1

    if isinstance(y_or_dict, dict):
        y = y_or_dict["y"]
        p0 = _parse_p0(y_or_dict)
    else:
        y = y_or_dict

    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob or (y == 0).all():
        return (np.nan,) * nparams, 0

    try:
        popt_, r2_ = _fit_internal(x, y)
    except RuntimeError:
        popt_, r2_ = (np.nan,) * nparams, 0
    return popt_, r2_


def _polyfit(y, x, deg, y_bounds=None, xp=None, rcond=None, w=None, eps=1e-8):
    def _fit_internal(_x, _y):
        popt = xp.polyfit(_x, _y, deg, rcond=rcond, w=w)

        residuals = _y - xp.polyval(popt, _x)
        ss_res = xp.sum(residuals ** 2)
        ss_tot = xp.sum((_y - xp.mean(_y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + eps))

        return popt, r_squared

    if xp in ["np", "numpy"]:
        xp = np
    elif xp in ["cp", "cupy"]:
        xp = cp
    else:
        raise ValueError("Cannot determine array module")

    nparams = deg + 1
    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob or (y == 0).all():
        return (xp.nan,) * nparams, 0

    try:
        popt_, r2_ = _fit_internal(x, y)
    except RuntimeError:
        popt_, r2_ = (xp.nan,) * nparams, 0
    return popt_, r2_


def _format_p0(p0, param_args, N):
    """Formats the p0 values for :func:`curve_fit`.

    Args:
        p0: The initialization values for the parameters.
        param_args (Sequence[str]): The parameter names.
        N (int): The number of sequences.

    Returns:
        p0_scalars (Dict[str, Optional[Number]]): The dictionary of
            all paramters and their scalar initializations. If parameter
            does not have a scalar value, the value for that key will be
            ``None``.
        p0_seq (Dict[str, Union[Sequence, ndarray]]): The dictionary of
            partial parameters that have a sequence-like structure.
    """
    nparams = len(param_args)

    if isinstance(p0, Number):
        p0 = (p0,) * nparams
    elif isinstance(p0, np.ndarray) and p0.ndim > 1:
        p0 = tuple(p0[..., i] for i in range(p0.shape[-1]))

    if isinstance(p0, (np.ndarray, Sequence)):
        if len(p0) != nparams:
            raise ValueError(f"`p0` has length {len(p0)} but function has {nparams} parameters")
        p0 = {param_args[i]: p0[i] for i in range(nparams)}
    elif isinstance(p0, Mapping):
        extra_keys = set(p0) - set(param_args)
        if len(extra_keys) > 0:
            raise ValueError(
                f"`p0` has unknown keys: {extra_keys}. "
                f"Function signature has parameters {param_args}."
            )
        p0_default = {p: 1.0 for p in param_args}
        # Update p0_default to keep the parameter keys in order.
        p0_default.update(p0)
        p0 = p0_default

    if p0 is not None:
        p0.update({k: 1.0 for k, v in p0.items() if v is None})

        p0_scalars = {k: v if not isinstance(v, np.ndarray) else None for k, v in p0.items()}
        p0_seq = {k: v for k, v in p0.items() if isinstance(v, np.ndarray)}
        for k, v in p0_seq.items():
            if len(v) != N:
                raise ValueError(f"Got {len(v)} values for param '{k}'. Expected {N}")
        if not p0_scalars:
            p0_scalars = None
        if not p0_seq:
            p0_scalars = tuple(p0_scalars.values())
            p0_seq = None
    else:
        p0_scalars, p0_seq = None, None

    return p0_scalars, p0_seq


def __fit_mono_exp__(x, y, p0=None):  # pragma: no cover
    def func(t, a, b):
        exp = np.exp(b * t)
        return a * exp

    warnings.warn(
        "__fit_mono_exp__ is deprecated since v0.0.12 and will no longer be "
        "supported in v0.0.13. Use `curve_fit` instead.",
        DeprecationWarning,
    )

    x = np.asarray(x)
    y = np.asarray(y)

    popt, _ = sop.curve_fit(func, x, y, p0=p0, maxfev=100, ftol=1e-5)

    residuals = y - func(x, popt[0], popt[1])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    return popt, r_squared


def __fit_monoexp_tc__(x, ys, tc0, show_pbar=False):  # pragma: no cover
    warnings.warn(
        "__fit_monoexp_tc__ is deprecated since v0.12 and will no longer be "
        "supported in v0.13. Use `curve_fit` instead.",
        DeprecationWarning,
    )

    p0 = (1.0, -1 / tc0)
    time_constants = np.zeros([1, ys.shape[-1]])
    r_squared = np.zeros([1, ys.shape[-1]])

    warned_negative = False
    for i in tqdm(range(ys.shape[1]), disable=not show_pbar):
        y = ys[..., i]
        if (y < 0).any() and not warned_negative:
            warned_negative = True
            warnings.warn(
                "Negative values found. Failure in monoexponential fit will result in np.nan"
            )

        # Skip any negative values or all values that are 0s
        if (y < 0).any() or (y == 0).all():
            continue

        try:
            params, r2 = __fit_mono_exp__(x, y, p0=p0)
            tc = 1 / abs(params[-1])
        except RuntimeError:
            tc, r2 = (np.nan, 0.0)

        time_constants[..., i] = tc
        r_squared[..., i] = r2

    return time_constants, r_squared
