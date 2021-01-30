from __future__ import print_function

import numpy as np

import matplotlib

matplotlib.use("TkAgg")


class IndexTracker:
    def __init__(self, ax, x):
        self.ax = ax
        self.im = None
        self.ind = 0
        self.x = x

    def onscroll(self, event):
        if event.button == "down":
            self.ind = min(self.ind + 1, self.num_slices - 1)
        elif event.button == "up":
            self.ind = max(self.ind - 1, 0)

        self.update()

    def update(self):
        x_im = self._x_normalized
        x_im = np.squeeze(x_im[:, :, self.ind, :])
        if self.im is None:
            self.im = self.ax.imshow(x_im, cmap="gray")
            self.ax.get_xaxis().set_ticks([])
            self.ax.get_yaxis().set_ticks([])

        self.im.set_data(x_im)
        self.ax.set_ylabel("slice %s" % (self.ind + 1))
        self.im.axes.figure.canvas.draw()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        assert type(value) is np.ndarray and (
            value.ndim == 3 or value.ndim == 4
        ), "image must be 3d (grayscale) or 4d (rgb) ndarray"
        if value.ndim == 3:
            value = value[..., np.newaxis]

        self._x = value
        self.num_slices = self._x.shape[2]
        self._x_normalized = self._x

        self.update()
