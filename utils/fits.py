import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize as sop

import defaults
from data_io.med_volume import MedicalVolume


class Fit(ABC):
    """Abstract class for fitting functionality"""

    @abstractmethod
    def fit(self):
        pass


class MonoExponentialFit(Fit):
    """Fit data using monoexponential fit of model A*exp(b*t)"""

    def __init__(self, ts, subvolumes, mask: MedicalVolume = None, bounds=(0, 100.0), tc0=30.0, decimal_precision=1):
        """
        :param ts: 1D list or numpy array of times corresponding to different subvolumes
        :param subvolumes: list of MedicalVolumes
        :param mask: a MedicalVolume mask of pixels to fit - speeds up fitting time
                        all pixels outside of mask are ignored
        :param bounds: tuple of [lower, upper) bound
        :param tc0: initial time constant guess (in milliseconds)
        :param decimal_precision: precision of rounding
        """
        assert (len(ts) == len(subvolumes))
        self.ts = ts

        assert (type(subvolumes) is list)
        for sv in subvolumes:
            assert type(sv) is MedicalVolume

        self.subvolumes = subvolumes

        self.mask = mask

        assert len(bounds) == 2, "Bounds should provide upper lower bound in format (lb, ub)"
        self.bounds = bounds

        self.tc0 = tc0
        self.decimal_precision = decimal_precision

    def fit(self):
        """Fit data used to initialize object
        :return: tuple of MedicalVolumes of time-constant estimate, r2 vals
        """
        svs = []
        msk = None

        subvolumes = self.subvolumes
        for sv in subvolumes[1:]:
            assert subvolumes[0].is_same_dimensions(sv), "Dimension mismatch"

        if self.mask:
            assert subvolumes[0].is_same_dimensions(self.mask), "Mask dimension mismatch"
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

        vals, r_squared = fit_monoexp_tc(self.ts, svs, self.tc0)

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        # All accepted values must meet an Rsquared threshold of DEFAULT_R2_THRESHOLD
        tc_map = map_unfiltered * (r_squared >= defaults.DEFAULT_R2_THRESHOLD)

        # Filter calculated values that are below limit bounds
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

    popt, _ = sop.curve_fit(func, x, y, p0=p0, maxfev=100)

    residuals = y - func(x, popt[0], popt[1])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / (ss_tot + __EPSILON__))

    return popt, r_squared


def fit_monoexp_tc(x, ys, tc0):
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
