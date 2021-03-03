import unittest

import numpy as np

from dosma.utils.device import Device, cpu_device

from ..util import requires_packages


class TestDevice(unittest.TestCase):
    def test_basic(self):
        assert Device(-1) == cpu_device
        assert Device("cpu") == cpu_device
        assert cpu_device.xp == np

        device = Device(-1)
        assert int(device) == -1

    @requires_packages("cupy")
    def test_cupy(self):
        import cupy as cp

        device = Device(0)
        assert device.cpdevice == cp.cuda.Device(0)
        assert device.type == "cuda"
        assert device.index == 0
        assert device.xp == cp
        assert int(device) == 0

        device = Device(cp.cuda.Device(0))
        assert device.cpdevice == cp.cuda.Device(0)
        assert device.type == "cuda"
        assert device.index == 0

    @requires_packages("sigpy")
    def test_sigpy(self):
        import sigpy as sp

        assert Device(-1) == sp.cpu_device
        assert Device(sp.cpu_device) == sp.cpu_device

        device = Device(-1)
        assert device.spdevice == sp.cpu_device

    @requires_packages("sigpy", "cupy")
    def test_sigpy_cupy(self):
        import sigpy as sp

        assert Device(0) == sp.Device(0)

        device = Device(0)
        assert device.spdevice == sp.Device(0)


if __name__ == "__main__":
    unittest.main()
