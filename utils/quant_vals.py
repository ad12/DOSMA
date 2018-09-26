from enum import Enum
import scipy.optimize as sop
import numpy as np

__EPSILON__ = 1e-8
class QuantitativeValue(Enum):
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


def fit_mono_exp(x, y, p0=None):
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

if __name__ == '__main__':
    print(type(QuantitativeValue.T1_RHO.name))

    for i in QuantitativeValue:
        print(i)