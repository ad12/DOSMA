import inspect
import multiprocessing as mp
import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy import optimize as sop
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dosma import defaults
from dosma.data_io.med_volume import MedicalVolume
from dosma.defaults import preferences

from typing import List, Sequence, Tuple

__all__ = ["MonoExponentialFit"]


class _Fit(ABC):
    """Abstract class for fitting quantitative values.
    """

    @abstractmethod
    def fit(self) -> Tuple[MedicalVolume, MedicalVolume]:
        """Fit quantitative values per pixel across multiple volumes.

        Pixels with errors in fitting are set to np.nan.

        Returns:
            tuple[MedicalVolume, MedicalVolume]: Quantitative value volume and r-squared goodness of fit volume.
        """
        pass


class MonoExponentialFit(_Fit):
    """Fit quantitative  using mono-exponential fit of model :math:`A*exp(b*t)`.

    Args:
        ts (:obj:`array-like`): 1D array of times in milliseconds (typically echo times) corresponding to different
            volumes.
        subvolumes (list[MedicalVolumes]): Volumes (in order) corresponding to times in `ts`.
        mask (:obj:`MedicalVolume`, optional): Mask of pixels to fit. If specified, pixels outside of mask region are
            ignored and set to `np.nan`. Speeds fitting time as fewer fits are required. Defaults to `None`.
        bounds (:obj:`tuple[float, float]`, optional): Upper and lower bound for quantitative values. Values outside
            those bounds will be set to `np.nan`. Defaults to `(0, 100.0)`.
        tc0 (:obj:`float`, optional): Initial time constant guess (in milliseconds). Defaults to `30.0`.
        decimal_precision (:obj:`int`, optional): Rounding precision after the decimal point. Defaults to `1`.
    """

    def __init__(
        self, ts: Sequence[float], subvolumes: List[MedicalVolume], mask: MedicalVolume = None,
        bounds: Tuple[float] = (0, 100.0), tc0: float = 30.0, decimal_precision: int = 1, verbose: bool = False,
        num_workers: int = 0,
    ):

        if (not isinstance(subvolumes, list)) or (not all([isinstance(sv, MedicalVolume) for sv in subvolumes])):
            raise TypeError("`subvolumes` must be list of MedicalVolumes.")

        if len(ts) != len(subvolumes):
            raise ValueError("`len(ts)`={:d}, but `len(subvolumes)`={:d}".format(len(ts), len(subvolumes)))
        self.ts = ts

        self.subvolumes = subvolumes

        if mask and not isinstance(mask, MedicalVolume):
            raise TypeError("`mask` must be a MedicalVolume")
        self.mask = mask
        self.verbose = verbose
        self.num_workers = num_workers

        orientation = self.subvolumes[0].orientation
        for sv in self.subvolumes[1:]:
            sv.reformat(orientation)

        if self.mask:
            self.mask.reformat(orientation)

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
            assert subvolumes[0].is_same_dimensions(self.mask,
                                                    defaults.AFFINE_DECIMAL_PRECISION), "Mask dimension mismatch"
            msk = self.mask.volume
            msk = msk.reshape(1, -1)

        original_shape = subvolumes[0].volume.shape
        affine = np.array(self.subvolumes[0].affine)

        for i in range(len(self.ts)):
            sv = subvolumes[i].volume

            svr = sv.reshape((1, -1))
            if msk is not None:
                svr = svr * msk

            svs.append(svr)

        svs = np.concatenate(svs)

        p0 = (1.0, -1 / self.tc0)
        popt, r_squared = curve_fit(
            monoexponential, self.ts, svs, self.bounds, p0=p0, 
            show_pbar=self.verbose, num_workers=self.num_workers,
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

        tc_map = np.around(tc_map, self.decimal_precision)

        time_constant_volume = MedicalVolume(tc_map, affine=affine)
        rsquared_volume = MedicalVolume(r_squared, affine=affine)

        return time_constant_volume, rsquared_volume


__EPSILON__ = 1e-8

def curve_fit(
    func, x, y, y_bounds=None, p0=None, maxfev=100, 
    ftol=1e-5, eps=1e-8, show_pbar=False, num_workers=0, 
    **kwargs,
):
    """
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
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.view(y.shape + (1,))
    N = y.shape[-1]

    func_args = inspect.getargspec(func).args
    nparams = len(func_args) - 2 if "self" in func_args else len(func_args) - 1
    
    if "bounds" not in kwargs:
        kwargs["maxfev"] = maxfev
    elif "max_nfev" not in kwargs:
        kwargs["max_nfev"] = maxfev
    
    num_workers = min(num_workers, N)
    fitter = partial(
        _curve_fit, x=x, func=func, y_bounds=y_bounds, p0=p0, 
        ftol=ftol, eps=eps, show_pbar=show_pbar, nparams=nparams, **kwargs,
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
            data = process_map(fitter, y.T, max_workers=num_workers, tqdm_class=tqdm)
        else:
            with mp.Pool(num_workers) as p:
                data = p.map(fitter, y.T)
        popts, r_squared = [x[0] for x in data], [x[1] for x in data]
    
    return np.stack(popts, axis=0), np.asarray(r_squared)


def _curve_fit(
    y, x, func, y_bounds=None, p0=None, 
    maxfev=100, ftol=1e-5, eps=1e-8, 
    show_pbar=False, nparams=None, **kwargs
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
        DeprecationWarning
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
        DeprecationWarning
    )

    p0 = (1.0, -1 / tc0)
    time_constants = np.zeros([1, ys.shape[-1]])
    r_squared = np.zeros([1, ys.shape[-1]])

    warned_negative = False
    for i in tqdm(range(ys.shape[1]), disable=not show_pbar):
        y = ys[..., i]
        if (y < 0).any() and not warned_negative:
            warned_negative = True
            warnings.warn("Negative values found. Failure in monoexponential fit will result in np.nan")

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
