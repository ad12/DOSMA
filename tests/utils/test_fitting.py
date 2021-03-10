import unittest

import numpy as np

from dosma.data_io.med_volume import MedicalVolume
from dosma.utils.fits import MonoExponentialFit, curve_fit, monoexponential

from .. import util


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
    def _generate_data(self, shape, x=None):
        b = np.random.rand(*shape)
        if x is None:
            x = np.asarray([0.5, 1.0, 2.0, 4.0])
        y = [MedicalVolume(monoexponential(t, 1.0, b), affine=np.eye(4)) for t in x]
        return x, y, b

    def test_basic(self):
        x, y, b = self._generate_data((10, 10, 20))
        t = 1 / np.abs(b)

        fitter = MonoExponentialFit(x, y, decimal_precision=8)
        t_hat = fitter.fit()[0]

        assert np.all(t_hat.volume - t <= 1e-8)

    def test_mask(self):
        x, y, b = self._generate_data((10, 10, 20))
        mask_arr = np.random.rand(*y[0].shape) > 0.5
        t = 1 / np.abs(b)

        mask = MedicalVolume(mask_arr, np.eye(4))
        fitter = MonoExponentialFit(x, y, mask, decimal_precision=8)
        t_hat = fitter.fit()[0]

        mask = mask.volume
        assert np.all(t_hat.volume[mask != 0] - t[mask != 0] <= 1e-8)

        with self.assertRaises(ValueError):
            fitter = MonoExponentialFit(x, y, mask_arr, decimal_precision=8)


if __name__ == "__main__":
    unittest.main()
