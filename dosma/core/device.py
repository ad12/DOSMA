"""Functions and classes for getting and setting computing devices.

"""
import numpy as np

from dosma.utils import env

if env.cupy_available():
    import cupy as cp
if env.sigpy_available():
    import sigpy as sp
if env.torch_available():
    import torch

__all__ = ["Device", "get_device", "to_device"]


class Device(object):
    """Device class.

    This class extends ``cupy.Device`` and can also be used to interface with
    ``torch.Device`` and ``sigpy.Device``. This class contains a device type
    ('cpu' or 'cuda') and optional device ordinal (i.e. the id) for the device type.
    This class can also be constructed using only the ordinal id, where id >= 0
    representing the id-th GPU, and id = -1 representing CPU. cupy must be installed to use GPUs.

    The array module for the corresponding device can be obtained via ``device.xp``.
    Similar to cupy.Device, the Device object can be used as a context:

        >>> device = Device(1)  # gpu 1
        >>> xp = device.xp  # xp is cupy.
        >>> with device:
        >>>     x = xp.array([1, 2, 3])
        >>>     x += 1


    Args:
        id_or_device (int or Device or cupy.cuda.Device): id > 0 represents
            the corresponding GPUs, and id = -1 represents CPU.

    Attributes:
        type (str): Device type. Either ``'cpu'`` or ``'cuda'``.
        index (int): index = -1 represents CPU, and others represents the id_th ordinances.

    Note:
        This class is heavily based on
        `sigpy.Device <https://sigpy.readthedocs.io/en/latest/generated/sigpy.Device.html>`_.

    """

    def __init__(self, id_or_device):
        _type, id = None, None
        if isinstance(id_or_device, int):
            id = id_or_device
        elif id_or_device == "cpu":
            _type, id = id_or_device, -1
        elif isinstance(id_or_device, Device):
            _type, id = id_or_device.type, id_or_device.id
        elif env.cupy_available() and isinstance(id_or_device, cp.cuda.Device):
            _type, id = "cuda", id_or_device.id
        elif env.sigpy_available() and isinstance(id_or_device, sp.Device):
            id = id_or_device.id
        elif env.torch_available() and isinstance(id_or_device, torch.device):
            _type, id = id_or_device.type, id_or_device.index
            if id is None:
                if _type == "cuda":
                    id = torch.cuda.current_device()
                elif _type == "cpu":
                    id = -1
                else:
                    raise ValueError(f"Unsupported device type: {_type}")
        else:
            raise ValueError(
                f"Accepts int, Device, cupy.cuda.Device, or torch.device" f"got {id_or_device}"
            )

        assert id >= -1
        if _type is None:
            _type = "cpu" if id == -1 else "cuda"

        cpdevice = None
        if id != -1:
            if env.cupy_available():
                cpdevice = cp.cuda.Device(id)
            else:
                raise ValueError("cupy not installed, but set device {}.".format(id))

        self._type = _type
        self._id = id
        self._cpdevice = cpdevice

    @property
    def id(self):
        """int: The device ordinal."""
        return self._id

    @property
    def type(self):
        """str: Type of device. Either ``"cpu"`` or ``"cuda"``."""
        return self._type

    @property
    def index(self):
        """int: Alias for ``self.id``."""
        return self.id

    @property
    def cpdevice(self):
        """cupy.Device: The equivalent ```cupy.Device```."""
        return self._cpdevice

    @property
    def ptdevice(self):
        """torch.device: The equivalent ```torch.device```."""
        if not env.torch_available():
            raise RuntimeError("`torch` not installed.")

        if self.id == -1:
            return torch.device("cpu")

        return torch.device(f"{self.type}:{self.id}")

    @property
    def spdevice(self):
        """sigpy.Device: The equivalent ```sigpy.Device```."""
        if not env.sigpy_available():
            raise RuntimeError("`sigpy` not installed.")
        if self.id >= 0 and self.type != "cuda":
            raise RuntimeError(f"sigpy.Device does not support type {self.type}")
        return sp.Device(self.id)

    @property
    def xp(self):
        """module: numpy or cupy module for the device."""
        if self.id == -1:
            return np
        return cp

    def use(self):
        """Use computing device.

        All operations after use() will use the device when using ``cupy``.
        """
        if self.id > 0:
            self.cpdevice.use()

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if other == -1:
            # other integers are not compared as self.type may be subject to change
            return self.id == other
        elif isinstance(other, Device):
            return self.type == other.type and self.id == other.id
        elif env.cupy_available() and isinstance(other, cp.cuda.Device):
            return self.type == "cuda" and self.id == other.id
        elif env.sigpy_available() and isinstance(other, sp.Device):
            try:
                return self.spdevice == other
            except RuntimeError:
                return False
        elif env.torch_available() and isinstance(other, torch.device):
            return self.ptdevice == other
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __enter__(self):
        if self.id == -1:
            return None

        return self.cpdevice.__enter__()

    def __exit__(self, *args):
        if self.id == -1:
            pass
        else:
            self.cpdevice.__exit__()

    def __repr__(self):
        if self.id == -1:
            return "Device(type='cpu')"

        return f"Device(type='{self.type}', index={self.id})"


cpu_device = Device(-1)


def get_array_module(array):
    """Gets an appropriate module from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if cupy is not available.

    Adapted from :mod:`sigpy`.

    Args:
        array: Input array.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on input.
    """
    if env.cupy_available():
        return cp.get_array_module(array)
    else:
        return np


def get_device(array):
    """Get Device from input array.

    Adapted from :mod:`sigpy`.

    Args:
        array (array): Array.

    Returns:
        Device.
    """
    if hasattr(array, "device"):
        return Device(array.device)
    else:
        return cpu_device


def to_device(input, device=cpu_device):
    """Move input to device. Does not copy if same device.

    Adapted from :mod:`sigpy`.

    Args:
        input (array): Input.
        device (int or Device or cupy.Device): Output device.

    Returns:
        array: Output array placed in device.
    """
    idevice = get_device(input)
    odevice = Device(device)

    if idevice == odevice:
        return input

    if odevice == cpu_device:
        with idevice:
            return input.get()
    else:
        with odevice:
            return cp.asarray(input)
