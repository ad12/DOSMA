from __future__ import print_function
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker():
    def __init__(self, ax, x):
        self.ax = ax
        self.im = None
        self.ind = 0
        self.x = x

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        x_im = self._x_normalized
        x_im = np.squeeze(x_im[:, :, self.ind, :])
        if self.im is None:
            self.im = self.ax.imshow(x_im, cmap='gray')

        self.im.set_data(x_im)
        self.ax.set_ylabel('slice %s' % (self.ind + 1))
        self.im.axes.figure.canvas.draw()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        assert type(value) is np.ndarray and (value.ndim == 3 or value.ndim == 4), "image must be 3d (grayscale) or 4d (rgb) ndarray"
        if value.ndim == 3:
            value = value[..., np.newaxis]

        self._x = value
        self.slices = self._x.shape[2]
        self._x_normalized = self._x

        self.update()

