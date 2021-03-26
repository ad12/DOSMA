import inspect
import multiprocessing as mp
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from scipy import optimize as sop
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from dosma import defaults
from dosma.data_io.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.utils.device import cpu_device, get_device

__all__ = ["CurveFitter", "MonoExponentialFit", "curve_fit", "monoexponential", "biexponential"]


class _Fit(ABC):
    """Abstract class for fitting quantitative values.
    """

    @abstractmethod
    def fit(self) -> Tuple[MedicalVolume, MedicalVolume]:
        """Fit quantitative values per pixel across multiple volumes.

        Pixels with errors in fitting are set to np.nan.

        Returns:
            tuple[MedicalVolume, MedicalVolume]: Quantitative value volume and
                r-squared goodness of fit volume.
        """
        pass


class CurveFitter:
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
        verbose: bool = False,
        **kwargs,
    ):
        func_name = func.__name__ if hasattr(func, "__name__") else type(func).__name__
        sig = inspect.signature(func)
        func_args = [p for p in sig.parameters]
        func_nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1

        if out_ufuncs is not None:
            if not isinstance(out_ufuncs, Callable) and not all(
                isinstance(ufunc, Callable) or ufunc is None for ufunc in out_ufuncs
            ):
                raise TypeError(
                    f"`out_ufuncs` must be callable or sequence of callables. Got {out_ufuncs}"
                )

            if isinstance(out_ufuncs, Sequence):
                if len(out_ufuncs) > func_nparams:
                    warnings.warn(
                        f"len(out_ufuncs)={len(out_ufuncs)}, but only {func_nparams} parameters. "
                        f"Extra ufuncs will be ignored."
                    )

        if out_bounds is not None:
            out_bounds = np.asarray(out_bounds)
            if out_bounds.shape[-1] != 2 or out_bounds.ndim > 2:
                raise ValueError("Invalid `out_bounds` - shape must be ([num_params,] 2)")
            if np.any(out_bounds[..., 0] > out_bounds[..., 1]):
                raise ValueError("Invalid `out_bounds` - lower bound must be <= upper bound")

        if isinstance(r2_threshold, str):
            if r2_threshold != "preferences":
                raise ValueError(
                    f"Invalid value r2_threshold='{r2_threshold}'. "
                    f"Expected `None`, a number, or 'preferences'."
                )
            r2_threshold = preferences.fitting_r2_threshold

        self._func = func
        self._func_name = func_name
        self.p0 = p0
        self.y_bounds = y_bounds
        self.out_ufuncs = out_ufuncs
        self.out_bounds = out_bounds
        self.r2_threshold = r2_threshold
        self.nan_to_num = nan_to_num
        self.num_workers = num_workers
        self.verbose = verbose
        self.kwargs = kwargs

    def _process_params(self, x):
        """
        Applies ``self.out_ufuncs`` and ``self.out_bounds``, in that order.
        Input is modified in place.

        Returns:
            ndarray: Values outside of ``self.out_bounds`` will be set to np.nan.
        """
        out_ufuncs = self.out_ufuncs
        out_bounds = self.out_bounds
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

        return x

    def fit(self, x, y: Sequence[MedicalVolume], mask: MedicalVolume = None):
        """
        Args:
            x (array-like): 1D array of independent variables corresponding to different ``y``.
            y (list[MedicalVolumes]): Dependent variable (in order) corresponding to values of
                independent variable ``x``. Data must be spatially aligned.
            mask (MedicalVolume, optional): If specified, only entries where ``mask > 0``
                will be fit.

        Returns:
            Tuple[MedicalVolume, MedicalVolume]: Tuple of fitted parameters (``popt``)
            and goodness of fit (``r2``) values. Last axis of fitted parameters
            corresponds to different parameters in order of appearance in ``self.func``.
        """
        svs = []
        msk = None

        if (not isinstance(y, (list, tuple))) or (
            not all([isinstance(_y, MedicalVolume) for _y in y])
        ):
            raise TypeError("`y` must be sequence of MedicalVolumes.")
        if any(_y.device != cpu_device for _y in y):
            raise RuntimeError("`y` must be on the CPU")
        if mask is not None and not isinstance(mask, MedicalVolume):
            raise TypeError("`mask` must be a MedicalVolume")

        x = np.asarray(x)
        if x.shape[-1] != len(y):
            raise ValueError(
                "Dimension mismatch: x.shape[-1]={:d}, but len(y)={:d}".format(x.shape[-1], len(y))
            )

        orientation = y[0].orientation
        y = [_y.reformat(orientation) for _y in y]

        if mask is not None:
            mask = mask.reformat_as(y[0])
            if not mask.is_same_dimensions(mask, defaults.AFFINE_DECIMAL_PRECISION):
                raise RuntimeError("`mask` and `y` dimension mismatch")
            msk = mask.volume
            msk = (msk > 0).astype(msk.dtype)
            msk = msk.reshape(1, -1)

        original_shape = y[0].shape
        for i in range(len(y)):
            sv = y[i].volume
            svr = sv.reshape((1, -1))
            if msk is not None:
                svr = svr * msk
            svs.append(svr)

        svs = np.concatenate(svs)

        popt, r_squared = curve_fit(
            self._func,
            x,
            svs,
            self.y_bounds,
            p0=self.p0,
            show_pbar=self.verbose,
            num_workers=self.num_workers,
            **self.kwargs,
        )

        popt = self._process_params(popt)
        if self.r2_threshold is not None:
            popt[(r_squared < self.r2_threshold)] = np.nan

        popt = popt.reshape(original_shape + popt.shape[-1:])
        r_squared = r_squared.reshape(original_shape)

        if self.nan_to_num is not None:
            popt = np.nan_to_num(popt, nan=self.nan_to_num, copy=False)

        popt = y[0]._partial_clone(volume=popt, headers=True)
        rsquared_volume = y[0]._partial_clone(volume=r_squared, headers=True)

        return popt, rsquared_volume

    def __str__(self) -> str:
        chunk_size = 1
        attrs = [
            "p0",
            "y_bounds",
            "out_bounds",
            "r2_threshold",
            "nan_to_num",
            "num_workers",
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


class MonoExponentialFit(_Fit):
    """Fit quantitative values using mono-exponential fit of model :math:`a*exp(-t/tc)`.

    Args:
        ts (:obj:`array-like`): 1D array of times in milliseconds (typically echo times)
            corresponding to different volumes.
        subvolumes (list[MedicalVolumes]): Volumes (in order) corresponding to times in `ts`.
        mask (:obj:`MedicalVolume`, optional): Mask of pixels to fit.
            If specified, pixels outside of mask region are ignored and set to ``np.nan``.
            Speeds fitting time as fewer fits are required.
        bounds (:obj:`tuple[float, float]`, optional): Upper and lower bound for quantitative
            values. Values outside those bounds will be set to ``np.nan``.
        tc0 (:obj:`float`, optional): Initial time constant guess (in milliseconds).
        decimal_precision (:obj:`int`, optional): Rounding precision after the decimal point.
    """

    def __init__(
        self,
        ts: Sequence[float],
        subvolumes: List[MedicalVolume],
        mask: MedicalVolume = None,
        bounds: Tuple[float] = (0, 100.0),
        tc0: float = 30.0,
        decimal_precision: int = 1,
        verbose: bool = False,
        num_workers: int = 0,
    ):

        if (not isinstance(subvolumes, list)) or (
            not all([isinstance(sv, MedicalVolume) for sv in subvolumes])
        ):
            raise TypeError("`subvolumes` must be list of MedicalVolumes.")

        if any(x.device != cpu_device for x in subvolumes):
            raise RuntimeError("All MedicalVolumes must be on the CPU")

        if len(ts) != len(subvolumes):
            raise ValueError(
                "`len(ts)`={:d}, but `len(subvolumes)`={:d}".format(len(ts), len(subvolumes))
            )
        self.ts = ts

        orientation = subvolumes[0].orientation
        subvolumes = [sv.reformat(orientation) for sv in subvolumes]
        self.subvolumes = subvolumes

        if mask and not isinstance(mask, MedicalVolume):
            raise TypeError("`mask` must be a MedicalVolume")
        self.mask = mask.reformat(orientation) if mask else None

        self.verbose = verbose
        self.num_workers = num_workers

        if len(bounds) != 2:
            raise ValueError("`bounds` should provide lower/upper bound in format (lb, ub)")
        self.bounds = bounds

        self.tc0 = tc0
        self.decimal_precision = decimal_precision

    def fit(self):
        svs = []
        msk = None

        subvolumes = self.subvolumes
        for sv in subvolumes[1:]:
            assert subvolumes[0].is_same_dimensions(sv), "Dimension mismatch within subvolumes"

        if self.mask:
            assert subvolumes[0].is_same_dimensions(
                self.mask, defaults.AFFINE_DECIMAL_PRECISION
            ), "Mask dimension mismatch"
            msk = self.mask.volume
            msk = (msk > 0).astype(msk.dtype)
            msk = msk.reshape(1, -1)

        original_shape = subvolumes[0].volume.shape

        for i in range(len(self.ts)):
            sv = subvolumes[i].volume

            svr = sv.reshape((1, -1))
            if msk is not None:
                svr = svr * msk

            svs.append(svr)

        svs = np.concatenate(svs)

        p0 = (1.0, -1 / self.tc0)
        popt, r_squared = curve_fit(
            monoexponential,
            self.ts,
            svs,
            y_bounds=None,
            p0=p0,
            show_pbar=self.verbose,
            num_workers=self.num_workers,
        )
        vals = 1 / np.abs(popt[:, 1])

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        # All accepted values must meet an r-squared threshold of `DEFAULT_R2_THRESHOLD`.
        tc_map = map_unfiltered * (r_squared >= preferences.fitting_r2_threshold)

        # Filter calculated values that are below limit bounds.
        tc_map[tc_map < self.bounds[0]] = np.nan
        tc_map = np.nan_to_num(tc_map)
        tc_map[tc_map > self.bounds[1]] = np.nan
        tc_map = np.nan_to_num(tc_map)

        if self.decimal_precision is not None:
            tc_map = np.around(tc_map, self.decimal_precision)

        time_constant_volume = self.subvolumes[0]._partial_clone(volume=tc_map, headers=True)
        rsquared_volume = self.subvolumes[0]._partial_clone(volume=r_squared, headers=True)

        return time_constant_volume, rsquared_volume


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
        p0 (Sequence, optional): Initial guess for the parameters (length N). If None, then
            the initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a ValueError is raised).
        maxfev (int, optional): Maximum number of function evaluations before the termination.
            If `bounds` argument for `scipy.optimize.curve_fit` is specified, this corresponds
            to the `max_nfev` in the least squares algorithm
        ftol (float): Tolerance for termination by the change of the cost function.
            See `scipy.optimize.least_squares` for more details.
        eps (float, optional): Epsilon for computing r-squared.
        show_pbar (bool, optional): If `True`, show progress bar. Note this can increase runtime
            slightly when using multiple workers.
        kwargs: Keyword args for `scipy.optimize.curve_fit`.
    """
    if (get_device(x) != cpu_device) or (get_device(y) != cpu_device):
        raise RuntimeError("`x` and `y` must be on CPU")

    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(y.shape + (1,))
    N = y.shape[-1]

    func_args = [p for p in inspect.signature(func).parameters]
    nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1

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
        p0=p0,
        ftol=ftol,
        eps=eps,
        show_pbar=show_pbar,
        nparams=nparams,
        **kwargs,
    )

    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob:
        warnings.warn("Out of bounds values found. Failure in fit will result in np.nan")

    popts = []
    r_squared = []
    if not num_workers:
        for i in tqdm(range(N), disable=not show_pbar):
            popt_, r2_ = fitter(y[:, i])
            popts.append(popt_)
            r_squared.append(r2_)
    else:
        if show_pbar:
            data = process_map(fitter, y.T, max_workers=num_workers)
        else:
            with mp.Pool(num_workers) as p:
                data = p.map(fitter, y.T)
        popts, r_squared = [x[0] for x in data], [x[1] for x in data]

    return np.stack(popts, axis=0), np.asarray(r_squared)


def _curve_fit(
    y,
    x,
    func,
    y_bounds=None,
    p0=None,
    maxfev=100,
    ftol=1e-5,
    eps=1e-8,
    show_pbar=False,
    nparams=None,
    **kwargs,
):
    def _fit_internal(_x, _y):
        popt, _ = sop.curve_fit(func, _x, _y, p0=p0, maxfev=maxfev, ftol=ftol, **kwargs)

        residuals = _y - func(_x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((_y - np.mean(_y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + eps))

        return popt, r_squared

    if nparams is None:
        func_args = inspect.getargspec(func).args
        nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1

    # import pdb; pdb.set_trace()
    oob = y_bounds is not None and ((y < y_bounds[0]).any() or (y > y_bounds[1]).any())
    if oob or (y == 0).all():
        return (np.nan,) * nparams, 0

    try:
        popt_, r2_ = _fit_internal(x, y)
    except RuntimeError:
        popt_, r2_ = (np.nan,) * nparams, 0
    return popt_, r2_


def monoexponential(x, a, b):
    """Function: :math:`f(x) = a * e^{b*x}`."""
    return a * np.exp(b * x)


def biexponential(x, a1, b1, a2, b2):
    """Function: :math:`f(x) = a1*e^{b1*x} + a2*e^{b2*x}`."""
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x)


def __fit_mono_exp__(x, y, p0=None):
    def func(t, a, b):
        exp = np.exp(b * t)
        return a * exp

    warnings.warn(
        "__fit_mono_exp__ is deprecated since v0.12 and will no longer be "
        "supported in v0.13. Use `curve_fit` instead.",
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


def __fit_monoexp_tc__(x, ys, tc0, show_pbar=False):
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
