import unittest

import numpy as np

from dosma.core.fitting import (
    CurveFitter,
    MonoExponentialFit,
    PolyFitter,
    curve_fit,
    monoexponential,
    polyfit,
)
from dosma.core.med_volume import MedicalVolume

from .. import util


def _generate_monoexp_data(shape=None, x=None, a=1.0, b=None):
    """Generate sample monoexponetial data.
    ``a=1.0``, ``b`` is randomly generated in interval [0.1, 1.1).

    The equation is :math:`y =  a * \\exp (b*x)`.
    """
    if b is None:
        b = np.random.rand(*shape) + 0.1
    else:
        shape = b.shape
    if x is None:
        x = np.asarray([0.5, 1.0, 2.0, 4.0])
    y = [MedicalVolume(monoexponential(t, a, b), affine=np.eye(4)) for t in x]
    return x, y, b


def _generate_affine(shape=None, x=None, a=None, b=1.0, as_med_vol=False):
    """Generate data of the form :math:`y = a*x + b`."""
    if a is None:
        a = np.random.rand(*shape) + 0.1
    else:
        shape = a.shape
    if x is None:
        x = np.asarray([0.5, 1.0, 2.0, 4.0])
    if b is None:
        b = np.random.rand(*shape)

    if as_med_vol:
        y = [MedicalVolume(a * t + b, affine=np.eye(4)) for t in x]
    else:
        y = [a * t + b for t in x]
    return x, y, a, b


def _linear(x, a):
    return a * x


def _generate_linear_data(shape=None, x=None, a=None):
    """Generate sample linear data.
    ``a`` is randomly generated in interval [0.1, 1.1).
    """
    if a is None:
        a = np.random.rand(*shape) + 0.1
    else:
        shape = a.shape
    if x is None:
        x = np.asarray([0.5, 1.0, 2.0, 4.0])
    y = [MedicalVolume(_linear(t, a), affine=np.eye(4)) for t in x]
    return x, y, a


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

        popt_mw, _ = curve_fit(
            monoexponential, x, ys, num_workers=util.num_workers(), show_pbar=True
        )
        assert np.allclose(popt, popt_mw)

    def test_p0(self):
        """Test different p0 configurations."""
        x = np.asarray([1, 2, 3, 4])
        num = 50
        ys = np.stack(
            [monoexponential(x, np.random.random(), np.random.random()) for _ in range(num)],
            axis=-1,
        )

        # Expected
        popt_1_1, _ = curve_fit(monoexponential, x, ys)
        popt_1_50, _ = curve_fit(monoexponential, x, ys, p0=(1.0, 50.0))

        # Scalar or empty value for each argument
        popt_seq, _ = curve_fit(monoexponential, x, ys, p0=(1.0, 1.0))
        assert np.allclose(popt_seq, popt_1_1)
        popt_seq, _ = curve_fit(monoexponential, x, ys, p0=(None, 1.0))
        assert np.allclose(popt_seq, popt_1_1)
        popt_seq, _ = curve_fit(monoexponential, x, ys, p0=(1.0, None))
        assert np.allclose(popt_seq, popt_1_1)

        # Dictionary of scalar values
        popt_dict, _ = curve_fit(monoexponential, x, ys, p0={"a": 1.0, "b": 1.0})
        assert np.allclose(popt_dict, popt_1_1)

        popt_dict, _ = curve_fit(monoexponential, x, ys, p0={"b": 50.0})
        assert np.allclose(popt_dict, popt_1_50)

        # Numpy array
        popt_arr, _ = curve_fit(monoexponential, x, ys, p0=np.ones((num, 2)))
        assert np.allclose(popt_arr, popt_1_1)

        p0 = np.stack([np.ones(num), 50 * np.ones(num)], axis=-1)
        popt_arr, _ = curve_fit(monoexponential, x, ys, p0=p0)
        assert np.allclose(popt_arr, popt_1_50)

        # Mix of array and scalar
        popt_arr, _ = curve_fit(monoexponential, x, ys, p0=[np.ones(num), 50])
        assert np.allclose(popt_arr, popt_1_50)

        # Multiprocessing
        popt_mw, _ = curve_fit(
            monoexponential, x, ys, p0=[np.ones(num), 50], num_workers=util.num_workers()
        )
        assert np.allclose(popt_mw, popt_1_50)

        popt_mw, _ = curve_fit(
            monoexponential,
            x,
            ys,
            p0=[np.ones(num), 50],
            num_workers=util.num_workers(),
            show_pbar=True,
        )
        assert np.allclose(popt_mw, popt_1_50)


class TestPolyFit(unittest.TestCase):
    """Test :func:`dosma.utils.fits.polyfit`."""

    def test_joint_optimization(self):
        x, y, a, b = _generate_affine((1000,), a=None, b=None, as_med_vol=False)
        popt_expected = np.polyfit(x, y, deg=1)
        popt, _ = polyfit(x, y, deg=1, num_workers=None)
        assert np.allclose(popt.T, popt_expected)
        assert np.allclose(popt[..., 0], a)
        assert np.allclose(popt[..., 1], b)

    def test_multiple_workers(self):
        x, y, _, _ = _generate_affine((1000,), a=None, b=None, as_med_vol=False)

        popt, _ = polyfit(x, y, deg=1, num_workers=0)
        popt_mw, _ = polyfit(x, y, deg=1, num_workers=util.num_workers())
        assert np.allclose(popt, popt_mw)

        popt_mw, _ = polyfit(x, y, deg=1, num_workers=util.num_workers(), show_pbar=True)
        assert np.allclose(popt, popt_mw)

    def test_polyfit_args(self):
        """Test that standard :func:`np.polyfit` args work."""
        x, y, _, _ = _generate_affine((1000,), a=None, b=None, as_med_vol=False)

        popt_exp, residuals_exp, rank_exp, singular_values_exp, rcond_exp = np.polyfit(
            x, y, deg=1, full=True
        )
        popt, _, residuals, rank, singular_values, rcond = polyfit(x, y, deg=1, full=True)

        assert np.allclose(popt, popt_exp.T)
        assert np.allclose(residuals, residuals_exp)
        assert np.allclose(rank, rank_exp)
        assert np.allclose(singular_values, singular_values_exp)
        assert np.allclose(rcond, rcond_exp)

        popt_exp, V_exp = np.polyfit(x, y, deg=1, cov=True)
        popt, _, V = polyfit(x, y, deg=1, cov=True)

        assert np.allclose(popt, popt_exp.T)
        assert np.allclose(V, V_exp)

    # @util.requires_packages("cupy")
    # def test_cupy(self):
    #     # TODO: Uncomment when cupy 9.0.0 is available.
    #     import cupy as cp

    #     x, y, _, _ = _generate_affine((1000,), a=None, b=None, as_med_vol=False)
    #     x_gpu = cp.asarray(x)
    #     y_gpu = cp.asarray(y)

    #     popt, _ = polyfit(x_gpu, y_gpu, deg=1)
    #     popt_expected = cp.polyfit(x_gpu, y_gpu, deg=1)
    #     assert cp.allclose(popt, popt_expected.T)


class TestMonoExponentialFit(unittest.TestCase):
    def test_basic(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        t = 1 / np.abs(b)

        fitter = MonoExponentialFit(decimal_precision=8)
        t_hat = fitter.fit(x, y)[0]
        assert np.allclose(t_hat.volume, t)

        with self.assertWarns(UserWarning):
            fitter = MonoExponentialFit(x, y, decimal_precision=8)
        t_hat = fitter.fit(x, y)[0]
        assert np.allclose(t_hat.volume, t)

        with self.assertRaises(ValueError):
            fitter = MonoExponentialFit(list(x) + [5], y)
        with self.assertRaises(TypeError):
            fitter = MonoExponentialFit(x, [_y.A for _y in y])
        with self.assertRaises(ValueError):
            fitter = MonoExponentialFit(x, y, tc0="a value")

    def test_headers(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        t = 1 / np.abs(b)
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:],
                fields={"StudyDescription": "Sample study", "EchoNumbers": idx},
            )

        fitter = MonoExponentialFit(decimal_precision=8)
        t_hat = fitter.fit(x, y)[0]

        assert np.allclose(t_hat.volume, t)
        assert t_hat.headers() is not None
        assert t_hat.headers().shape == (1, 1, 20)
        for h in t_hat.headers().flatten():
            assert h.get("StudyDescription") == "Sample study"

    def test_mask(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        t = 1 / np.abs(b)

        mask = MedicalVolume(mask_arr, np.eye(4))
        fitter = MonoExponentialFit(decimal_precision=8)
        t_hat = fitter.fit(x, y, mask)[0]

        mask = mask.volume
        assert np.allclose(t_hat.volume[mask != 0], t[mask != 0])

        fitter2 = MonoExponentialFit(decimal_precision=8)
        t_hat2 = fitter2.fit(x, y, mask_arr)[0]
        assert np.allclose(t_hat2.volume, t_hat.volume)

        with self.assertWarns(UserWarning):
            fitter3 = MonoExponentialFit(mask=mask, decimal_precision=8)
        t_hat3 = fitter3.fit(x, y)[0]
        assert np.allclose(t_hat3.volume, t_hat.volume)

    def test_polyfit_initialization(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))
        t = 1 / np.abs(b)

        fitter = MonoExponentialFit(tc0="polyfit", decimal_precision=8)
        t_hat = fitter.fit(x, y)[0]
        assert np.allclose(t_hat.volume, t)

        # Test fitting still works even if some values are 0.
        # The values will not be accurate, but other pixel values should be.
        x, y, b = _generate_monoexp_data((10, 10, 20))
        t = 1 / np.abs(b)
        mask_arr = np.zeros(y[0].shape, dtype=bool)
        mask_arr[:5, :5] = 1
        y[0][mask_arr] = 0

        fitter = MonoExponentialFit(tc0="polyfit", decimal_precision=8)
        t_hat = fitter.fit(x, y)[0]
        assert np.allclose(t_hat.volume[mask_arr == 0], t[mask_arr == 0])


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
        assert np.all(np.isnan(a_hat.volume[mask_arr == 0]))
        assert np.all(np.isnan(b_hat.volume[mask_arr == 0]))

        fitter = CurveFitter(monoexponential)
        popt = fitter.fit(x, y, mask=mask_arr)[0]
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume[mask_arr != 0], 1.0)
        assert np.allclose(b_hat.volume[mask_arr != 0], b[mask_arr != 0])

        with self.assertRaises(TypeError):  # noqa B908
            fitter = CurveFitter(monoexponential)
            popt = fitter.fit(x, y, mask="foo")[0]

        with self.assertRaises(RuntimeError):  # noqa B908
            mask_incorrect_shape = np.random.rand(5, 5, 5) > 0.5
            fitter = CurveFitter(monoexponential)
            popt = fitter.fit(x, y, mask=mask_incorrect_shape)[0]

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

        # Bounds should only be lower and upper.
        with self.assertRaises(ValueError):
            fitter = CurveFitter(monoexponential, out_bounds=[(0, 0.5, 1.0)])

        # Check lower bound < upper bound
        with self.assertRaises(ValueError):
            fitter = CurveFitter(monoexponential, out_bounds=[(1.2, 0)])

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

        with self.assertRaises(TypeError):
            fitter = CurveFitter(monoexponential, out_ufuncs=[None, 5])

        with self.assertWarns(UserWarning):
            fitter = CurveFitter(monoexponential, out_ufuncs=[None, ufunc, ufunc])

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

        fitter = MonoExponentialFit(tc0=30.0, bounds=(0, 100), decimal_precision=8)
        t_hat_mef = fitter.fit(x, y)[0]

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

    def test_headers(self):
        """Test curve fitter does not fail with volumes with headers."""
        x, y, b = _generate_monoexp_data((10, 10, 20, 4))
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:], fields={"EchoNumbers": idx}
            )

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)
        assert b_hat.headers() is not None
        assert b_hat.headers().shape == (1, 1, 20, 4)

        # Test header with single parameter to fit
        x, y, a = _generate_linear_data((10, 10, 20, 4))
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:], fields={"EchoNumbers": idx}
            )

        fitter = CurveFitter(_linear)
        popt, _ = fitter.fit(x, y)
        a_hat = popt[..., 0]

        assert np.allclose(a_hat.volume, a)

        # Test not copying headers
        x, y, b = _generate_monoexp_data((10, 10, 20, 4))
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:], fields={"EchoNumbers": idx}
            )

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y, copy_headers=False)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)
        assert a_hat.headers() is None
        assert b_hat.headers() is None

    def test_p0(self):
        x, y, b = _generate_monoexp_data((10, 10, 20))

        fitter = CurveFitter(monoexponential, p0=(1.0, b))
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(monoexponential, p0={"a": 1.0, "b": b})
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(
            monoexponential, p0={"a": 1.0, "b": MedicalVolume(b, affine=y[0].affine)}
        )
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y, p0=(1.0, b))
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y, p0={"a": 1.0, "b": b})
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y, p0={"a": 1.0, "b": MedicalVolume(b, affine=y[0].affine)})
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        fitter = CurveFitter(monoexponential)
        p0 = np.stack(
            [
                MedicalVolume(np.ones(b.shape), affine=y[0].affine),
                MedicalVolume(b, affine=y[0].affine),
            ],
            axis=-1,
        )
        popt, _ = fitter.fit(x, y, p0=p0)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)

        # Test combination of setting p0 with a volume and using a mask.
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        mask = MedicalVolume(mask_arr, y[0].affine)

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(
            x,
            y,
            p0={"a": 1.0, "b": MedicalVolume(b, affine=y[0].affine)},
            mask=mask,
        )
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume[mask_arr != 0], 1.0)
        assert np.allclose(b_hat.volume[mask_arr != 0], b[mask_arr != 0])
        assert np.all(np.isnan(a_hat.volume[mask_arr == 0]))
        assert np.all(np.isnan(b_hat.volume[mask_arr == 0]))

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(
            x,
            y,
            p0=(1.0, b),
            mask=mask_arr,
        )
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.volume[mask_arr != 0], 1.0)
        assert np.allclose(b_hat.volume[mask_arr != 0], b[mask_arr != 0])
        assert np.all(np.isnan(a_hat.volume[mask_arr == 0]))
        assert np.all(np.isnan(b_hat.volume[mask_arr == 0]))

    def test_str(self):
        fitter = CurveFitter(
            monoexponential,
            p0=(1.0, -1 / 30),
            out_ufuncs=[None, lambda x: 1 / np.abs(x)],
            out_bounds=(0, 100),
            nan_to_num=0,
        )
        _ = str(fitter)


class TestPolyFitter(unittest.TestCase):
    """Tests for ``dosma.utils.fits.PolyFitter``."""

    def test_basic(self):
        x, y, a, b = _generate_affine((10, 10, 20), as_med_vol=True)
        fitter = PolyFitter(deg=1)
        popt, r2 = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, a)
        assert np.allclose(b_hat.volume, b)

        assert np.all(popt.affine == y[0].affine)
        assert np.all(r2.affine == y[0].affine)

    def test_mask(self):
        x, y, a, b = _generate_affine((10, 10, 20), as_med_vol=True)
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        mask = MedicalVolume(mask_arr, y[0].affine)

        fitter = PolyFitter(deg=1)
        popt = fitter.fit(x, y, mask=mask)[0]
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume[mask_arr != 0], a[mask_arr != 0])
        assert np.allclose(b_hat.volume[mask_arr != 0], b)

        fitter = PolyFitter(deg=1)
        popt = fitter.fit(x, y, mask=mask_arr)[0]
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume[mask_arr != 0], a[mask_arr != 0])
        assert np.allclose(b_hat.volume[mask_arr != 0], b)

    def test_headers(self):
        """Test curve fitter does not fail with volumes with headers."""
        x, y, a, b = _generate_affine((10, 10, 20, 4), as_med_vol=True)
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:], fields={"EchoNumbers": idx}
            )

        fitter = PolyFitter(deg=1)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, a)
        assert np.allclose(b_hat.volume, b)

        # Test header with single parameter to fit
        x, y, a, _ = _generate_affine((10, 10, 20, 4), b=0.0, as_med_vol=True)

        fitter = PolyFitter(deg=1)
        popt, _ = fitter.fit(x, y)
        a_hat = popt[..., 0]

        assert np.allclose(a_hat.volume, a)

        # Test not copying headers
        x, y, b = _generate_monoexp_data((10, 10, 20, 4))
        for idx, _y in enumerate(y):
            _y._headers = util.build_dummy_headers(
                (1, 1) + _y.shape[2:], fields={"EchoNumbers": idx}
            )

        fitter = CurveFitter(monoexponential)
        popt, _ = fitter.fit(x, y, copy_headers=False)
        a_hat, b_hat = popt[..., 0], popt[..., 1]

        assert np.allclose(a_hat.volume, 1.0)
        assert np.allclose(b_hat.volume, b)
        assert a_hat.headers() is None
        assert b_hat.headers() is None

    def test_nan_to_num(self):
        shape = (10, 10, 20)
        a = np.ones(shape)
        a[5:] = 1.5
        b = np.random.rand(*shape) + 0.1
        b[:5] = 1.5

        x, y, _, _ = _generate_affine(a=a, b=b, as_med_vol=True)

        out_bounds = (0, 1.2)
        fitter = PolyFitter(deg=1, out_bounds=out_bounds, nan_to_num=0.0)
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat[:5].volume, 1.0) and np.allclose(a_hat[5:].volume, 0.0)
        assert np.allclose(b_hat[5:].volume, b[5:]) and np.allclose(b_hat[:5].volume, 0.0)

    def test_out_ufuncs(self):
        x, y, a, b = _generate_affine((10, 10, 20, 4), as_med_vol=True)
        fitter = PolyFitter(deg=1, out_ufuncs=[lambda x: x + 1, lambda x: x + 2])
        popt, _ = fitter.fit(x, y)
        a_hat, b_hat = popt[..., 0], popt[..., 1]
        assert np.allclose(a_hat.A, a + 1)
        assert np.allclose(b_hat.A, b + 2)

    def test_str(self):
        fitter = PolyFitter(deg=2, rcond=0.5, y_bounds=(0, 200), r2_threshold=0.9, chunksize=1000)
        _ = str(fitter)


if __name__ == "__main__":
    unittest.main()
