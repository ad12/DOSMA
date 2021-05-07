import unittest

import numpy as np

from dosma.core.device import Device, cpu_device, get_device, to_device
from dosma.core.med_volume import MedicalVolume

from ..util import requires_packages


class TestDevice(unittest.TestCase):
    def test_basic(self):
        assert Device(-1) == cpu_device
        assert Device("cpu") == cpu_device
        assert cpu_device.xp == np

        device = Device(-1)
        assert int(device) == -1
        assert device.index == -1
        assert device.id == -1
        assert device == -1
        assert device.cpdevice is None

        device2 = Device(-1)
        assert device2 == device

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
        assert device == sp.cpu_device
        assert device.spdevice == sp.cpu_device

    @requires_packages("sigpy", "cupy")
    def test_sigpy_cupy(self):
        import sigpy as sp

        assert Device(0) == sp.Device(0)

        device = Device(0)
        assert device.spdevice == sp.Device(0)

    @requires_packages("torch")
    def test_torch(self):
        import torch

        pt_device = torch.device("cpu")

        assert Device(pt_device) == cpu_device

        dm_device = Device(-1)
        assert dm_device == pt_device
        assert dm_device.ptdevice == pt_device

    def test_to_device(self):
        arr = np.ones((3, 3, 3))
        mv = MedicalVolume(arr, affine=np.eye(4))

        arr2 = to_device(arr, -1)
        assert get_device(arr2) == cpu_device

        mv2 = to_device(mv, -1)
        assert get_device(mv2) == cpu_device


if __name__ == "__main__":
    unittest.main()
