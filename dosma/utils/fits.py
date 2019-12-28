import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize as sop

from dosma import defaults
from dosma.data_io.med_volume import MedicalVolume
from dosma.defaults import preferences

from typing import List, Tuple, Union

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

    def __init__(self, ts: Union[List, np.ndarray][float], subvolumes: List[MedicalVolume], mask: MedicalVolume = None,
                 bounds: Tuple[float, float] = (0, 100.0), tc0: float = 30.0, decimal_precision: int = 1):

        if (not isinstance(subvolumes, list)) or (not all([isinstance(sv, MedicalVolume) for sv in subvolumes])):
            raise TypeError("`subvolumes` must be list of MedicalVolumes.")

        if len(ts) != len(subvolumes):
            raise ValueError("`len(ts)`={:d}, but `len(subvolumes)`={:d}".format(len(ts), len(subvolumes)))
        self.ts = ts

        self.subvolumes = subvolumes

        if mask and not isinstance(mask, MedicalVolume):
            raise TypeError("`mask` must be a MedicalVolume")
        self.mask = mask

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

        vals, r_squared = __fit_monoexp_tc__(self.ts, svs, self.tc0)

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        # All accepted values must meet an r-squared threshold of `DEFAULT_R2_THRESHOLD`.
        tc_map = map_unfiltered * (r_squared >= preferences.fitting_r2_threshold)

        # Filter calculated values that are below limit bounds.
        tc_map[tc_map <= self.bounds[0]] = np.nan
        tc_map = np.nan_to_num(tc_map)
        tc_map[tc_map > self.bounds[1]] = np.nan
        tc_map = np.nan_to_num(tc_map)

        tc_map = np.around(tc_map, self.decimal_precision)

        time_constant_volume = MedicalVolume(tc_map, affine=affine)
        rsquared_volume = MedicalVolume(r_squared, affine=affine)

        return time_constant_volume, rsquared_volume


__EPSILON__ = 1e-8


def __fit_mono_exp__(x, y, p0=None):
    def func(t, a, b):
        exp = np.exp(b * t)
        return a * exp

    x = np.asarray(x)
    y = np.asarray(y)

    popt, _ = sop.curve_fit(func, x, y, p0=p0, maxfev=100, ftol=1e-5)

    residuals = y - func(x, popt[0], popt[1])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    return popt, r_squared


def __fit_monoexp_tc__(x, ys, tc0):
    p0 = (1.0, -1 / tc0)
    time_constants = np.zeros([1, ys.shape[-1]])
    r_squared = np.zeros([1, ys.shape[-1]])

    warned_negative = False
    for i in range(ys.shape[1]):
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
