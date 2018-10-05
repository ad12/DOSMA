from enum import Enum
import scipy.optimize as sop
import numpy as np
import glob
import warnings


__EPSILON__ = 1e-8
__R_SQUARED_THRESHOLD__ = 0.9


class QuantitativeValues(Enum):
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


def get_qv(id):
    for qv in QuantitativeValues:
        if qv.name.lower() == id or qv.value == id:
            return qv


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
    p0 = (1.0, -1/tc0)
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


if __name__ == '__main__':
    print(type(QuantitativeValues.T1_RHO.name))
    print(QuantitativeValues.T1_RHO.value == 1)