import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize as sop

import defaults
from med_objects.med_volume import MedicalVolume


class Fit(ABC):
    @abstractmethod
    def fit(self):
        pass


class MonoExponentialFit(Fit):
    def __init__(self, ts, subvolumes, mask=None, bounds=(0, 100.0), tc0=30.0, decimal_precision=1):
        assert (len(ts) == len(subvolumes))
        self.ts = ts

        assert (type(subvolumes) is list)
        for sv in subvolumes:
            assert type(sv) is MedicalVolume

        self.subvolumes = subvolumes

        if mask is not None:
            assert (type(mask) is MedicalVolume)
        self.mask = mask

        assert (type(bounds) is tuple and len(bounds) == 2)
        self.bounds = bounds

        self.tc0 = tc0
        self.decimal_precision = decimal_precision

    def fit(self):
        original_shape = None
        svs = []
        msk = None
        if self.mask:
            msk = self.mask.volume
            msk = msk.reshape(1, -1)

        pixel_spacing = self.subvolumes[0].pixel_spacing

        for i in range(len(self.ts)):
            sv = self.subvolumes[i].volume

            if original_shape is None:
                original_shape = sv.shape
            else:
                assert (sv.shape == original_shape)

            svr = sv.reshape((1, -1))
            if msk is not None:
                svr = svr * msk

            svs.append(svr)

        svs = np.concatenate(svs)

        vals, r_squared = fit_monoexp_tc(self.ts, svs, self.tc0)

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        tc_map = map_unfiltered * (r_squared > defaults.DEFAULT_R2_THRESHOLD)

        # Filter calculated T1-rho values that are below 0ms and over 100ms
        tc_map[tc_map <= self.bounds[0]] = np.nan
        tc_map = np.nan_to_num(tc_map)
        tc_map[tc_map > self.bounds[1]] = np.nan
        tc_map = np.nan_to_num(tc_map)

        tc_map = np.around(tc_map, self.decimal_precision)

        return MedicalVolume(tc_map, pixel_spacing), MedicalVolume(r_squared, pixel_spacing)


__EPSILON__ = 1e-8


def __fit_mono_exp__(x, y, p0=None):
    def func(t, a, b):
        exp = np.exp(b * t)
        return a * exp

    x = np.asarray(x)
    y = np.asarray(y)

    popt, _ = sop.curve_fit(func, x, y, p0=p0, maxfev=1000)

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
            warnings.warn("Negative values found. Failure in monoexponential fit will result in t1_rho=np.nan")

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
