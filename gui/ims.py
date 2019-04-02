import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from tkinter import filedialog, messagebox, Radiobutton, IntVar
import tkinter as tk
from tkinter import ttk

from gui.im_viewer import IndexTracker
from data_io import format_io_utils as fio_utils
from data_io.nifti_io import __NIFTI_EXTENSIONS__
from data_io.dicom_io import __DICOM_EXTENSIONS__
import os
from data_io.orientation import SAGITTAL, CORONAL, AXIAL
from skimage.measure import label
from skimage.color import label2rgb
from copy import deepcopy
LARGE_FONT = ("Verdana", 12)


class DosmaViewer(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        # button = ttk.Button(self, text="Visit Page 1",
        #                     command=lambda: controller.show_frame(PageOne))
        # button.pack()
        #
        # button2 = ttk.Button(self, text="Visit Page 2",
        #                      command=lambda: controller.show_frame(PageTwo))
        # button2.pack()

        button3 = ttk.Button(self, text="Image Viewer",
                             command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                             command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                             command=lambda: controller.show_frame(PageOne))
        button2.pack()


class PageThree(tk.Frame):
    SUPPORTED_FORMATS = (('nifti files', '*.nii\.gz'), ('dicom files', '*.dcm'))
    __base_filepath = '../'

    _ORIENTATIONS = [('sagittal', SAGITTAL), ('coronal', CORONAL), ('axial', AXIAL)]

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self._im_display = None
        self.binding_vars = dict()
        fig, ax = plt.subplots(1, 1)
        X = np.random.rand(20, 20, 40)

        self.tracker = IndexTracker(ax, X)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.mpl_connect('scroll_event', self.tracker.onscroll)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.im = None
        self.mask = None
        self._im_display = None

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(side=tk.BOTTOM, anchor="sw")

        button2 = ttk.Button(self, text='Load main image', command=self.load_volume_callback)
        button2.pack()

        button3 = ttk.Button(self, text='Load mask', command=self.load_mask_callback)
        button3.pack()

        self.init_reformat_display()

    def __reformat_callback(self, *args):
        self.im_update()

    def init_reformat_display(self):
        orientation_var = IntVar(0)
        orientation_var.trace_add('write', self.__reformat_callback)
        count = 0
        for text, value in self._ORIENTATIONS:
            b = Radiobutton(self, text=text, variable=orientation_var, value=count)
            b.pack(side=tk.TOP, anchor='w')
            count += 1
        self._orientation = orientation_var

    def load_volume_callback(self):
        im = self.load_volume()
        if not im:
            return
        self.im = im
        self.mask = None

        self.im_update()

    def load_mask_callback(self):
        if not self.im:
            messagebox.showerror('Loading mask failed', 'Main image must be loaded prior to mask')
            return

        mask = self.load_volume("Load mask")
        mask.reformat(self.im.orientation)
        try:
            self.__verify_mask_size(self.im.volume, mask.volume)
        except Exception as e:
            messagebox.showerror('Loading mask failed', str(e))
            return

        self.mask = mask
        self.im_update()

    def __verify_mask_size(self, im: np.ndarray, mask: np.ndarray):
        if mask.ndim != 3:
            raise ValueError('Dimension mismatch. Mask must be 3D')
        if im.shape != mask.shape:
            raise ValueError('Dimension mismatch. Image of shape %s, but mask of shape %s' % (str(im.shape),
                                                                                              str(mask.shape)))

    def im_update(self):
        orientation = self.orientation
        self.im.reformat(orientation)
        im = self.im.volume
        im = im / np.max(im)
        if self.mask:
            self.mask.reformat(orientation)
            label_image = label(self.mask.volume)
            im = self.__labeltorgb_3d__(im, label_image, 0.3)

        self.im_display = im

    def __labeltorgb_3d__(self, im: np.ndarray, labels:np.ndarray, alpha: float=0.3):
        im_rgb = np.zeros(im.shape + (3,))  # rgb channel
        for s in range(im.shape[2]):
            im_slice = im[..., s]
            labels_slice = labels[..., s]
            im_rgb[..., s, :] = label2rgb(labels_slice, image=im_slice, bg_label=0, alpha=alpha)
        return im_rgb

    def load_volume(self, title='Select volume file(s)'):
        files = filedialog.askopenfilenames(initialdir=self.__base_filepath, title=title)
        if len(files) == 0:
            return

        filepath = files[0]
        self.__base_filepath = os.path.dirname(filepath)

        if filepath.endswith('.dcm'):
            filepath = os.path.dirname(filepath)

        im = fio_utils.generic_load(filepath, 1)

        return im

    @property
    def orientation(self):
        ind = self._orientation.get()
        return self._ORIENTATIONS[ind][1]

    @property
    def im_display(self):
        return self._im_display

    @im_display.setter
    def im_display(self, value):
        self._im_display = value
        self.tracker.x = self._im_display


app = DosmaViewer()
app.mainloop()