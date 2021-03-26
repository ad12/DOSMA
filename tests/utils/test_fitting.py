import unittest

import numpy as np

from dosma.data_io.med_volume import MedicalVolume
from dosma.utils.fits import CurveFitter, MonoExponentialFit, curve_fit, monoexponential

from .. import util


def _generate_monoexp_data(shape=None, x=None, a=1.0, b=None):
    """Generate sample monoexponetial data.
    ``a=1.0``, ``b`` is randomly generated in interval [0.1, 1.1).
    """
    if b is None:
        b = np.random.rand(*shape) + 0.1
    else:
        shape = b.shape
    if x is None:
        x = np.asarray([0.5, 1.0, 2.0, 4.0])
    y = [MedicalVolume(monoexponential(t, a, b), affine=np.eye(4)) for t in x]
    return x, y, b


class TestCurveFit(unittest.TestCase):
    def test_multiple_workers(self):
        x = np.asarray([1, 2, 3, 4])
        ys = np.stack(
            [monoexponential(x, np.random.random(), np.random.random()) for _ in range(1000)],
            axis=-1,
        )
        popt, _ = curve_fit(monoexponential, x, ys)
        popt_mw, _ = curve_fit(monoexponential, x, ys, num_workers=util.num_workers())
        assert np.allclose(popt, popt_mw)


class TestMonoExponentialFit(unittest.TestCase):
    def test_basic(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        t = 1 / np.abs(b)

        fitter = MonoExponentialFit(x, y, decimal_precision=8)
        t_hat = fitter.fit()[0]

        assert np.allclose(t_hat.volume, t)

    def test_mask(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        t = 1 / np.abs(b)

        mask = MedicalVolume(mask_arr, np.eye(4))
        fitter = MonoExponentialFit(x, y, mask, decimal_precision=8)
        t_hat = fitter.fit()[0]

        mask = mask.volume
        assert np.allclose(t_hat.volume[mask != 0], t[mask != 0])

        with self.assertRaises(ValueError):
            fitter = MonoExponentialFit(x, y, mask_arr, decimal_precision=8)


class TestCurveFitter(unittest.TestCase):
    """Tests for ``dosma.utils.fits.CurveFitter``."""

    def test_basic(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        fitter = CurveFitter(monoexponential)
        popt, r2 = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        assert np.all(popt.affine == y[0].affine)
        assert np.all(r2.affine == y[0].affine)

    def test_mask(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        mask = MedicalVolume(mask_arr, y[0].affine)

        fitter = CurveFitter(monoexponential)
        popt = fitter.fit(x, y, mask=mask)[0]
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume[mask_arr != 0], 1.0)
        assert np.allclose(b_hat.volume[mask_arr != 0], b[mask_arr != 0])

    def test_bounds(self):
        shape = (10, 10, 20)
        a = np.ones(shape)
        a[5:] = 1.5
        b = np.random.rand(*shape) + 0.1
        b[:5] = 1.5

        x, y, _ = _generate_monoexp_data(a=a, b=b)

        # Bounds for all parameters
        out_bounds = (0, 1.2)
        fitter = CurveFitter(monoexponential, out_bounds=out_bounds)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat[:5].volume, 1.0) and np.all(np.isnan(a_hat[5:].volume))
        assert np.allclose(b_hat[5:].volume, b[5:]) and np.all(np.isnan(b_hat[:5].volume))

        # Bounds only for second parameter
        out_bounds = [(-np.inf, np.inf), (0, 1.2)]
        fitter = CurveFitter(monoexponential, out_bounds=out_bounds)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, a)
        assert np.allclose(b_hat[5:].volume, b[5:]) and np.all(np.isnan(b_hat[:5].volume))

        # Bounds only for first parameter
        out_bounds = [(0, 1.2)]
        fitter = CurveFitter(monoexponential, out_bounds=out_bounds)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat[:5].volume, 1.0) and np.all(np.isnan(a_hat[5:].volume))
        assert np.allclose(b_hat.volume, b)

    def test_out_ufuncs(self):
        shape = (10, 10, 20)
        a = -1
        b = np.random.rand(*shape) - 1.1  # all negative values
        x, y, _ = _generate_monoexp_data(a=a, b=b)

        ufunc = lambda x: 2 * np.abs(x) + 5  # noqa: E731

        fitter = CurveFitter(monoexponential, out_ufuncs=ufunc)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, ufunc(a))
        assert np.allclose(b_hat.volume, ufunc(b))

        fitter = CurveFitter(monoexponential, out_ufuncs=[None, ufunc])
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, a)
        assert np.allclose(b_hat.volume, ufunc(b))

        fitter = CurveFitter(monoexponential, out_ufuncs=[ufunc])
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, ufunc(a))
        assert np.allclose(b_hat.volume, b)

    def test_nan_to_num(self):
        shape = (10, 10, 20)
        a = np.ones(shape)
        a[5:] = 1.5
        b = np.random.rand(*shape) + 0.1
        b[:5] = 1.5

        x, y, _ = _generate_monoexp_data(a=a, b=b)

        out_bounds = (0, 1.2)
        fitter = CurveFitter(monoexponential, out_bounds=out_bounds, nan_to_num=0.0)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat[:5].volume, 1.0) and np.allclose(a_hat[5:].volume, 0.0)
        assert np.allclose(b_hat[5:].volume, b[5:]) and np.allclose(b_hat[:5].volume, 0.0)

    def test_matches_monoexponential_fit(self):
        """Match functionality of ``MonoexponentialFit`` using ``CurveFitter``."""
        x, y, _ = _generate_monoexp_data((10, 10, 20))

        fitter = MonoExponentialFit(x, y, tc0=30.0, bounds=(0, 100), decimal_precision=8)
        t_hat_mef = fitter.fit()[0]

        fitter = CurveFitter(
            monoexponential,
            p0=(1.0, -1 / 30),
            out_ufuncs=[None, lambda x: 1 / np.abs(x)],
            out_bounds=(0, 100),
            nan_to_num=0,
        )
        t_hat_cf = fitter.fit(x, y)[0][..., 1]
        t_hat_cf = np.round(t_hat_cf, decimals=8)

        assert np.allclose(t_hat_mef.volume, t_hat_cf.volume)


if __name__ == "__main__":
    unittest.main()
