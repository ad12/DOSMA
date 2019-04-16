import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

from tkinter import filedialog, messagebox, Radiobutton, IntVar
import tkinter as tk
from tkinter import ttk

from gui.im_viewer import IndexTracker
from data_io import format_io_utils as fio_utils
import os
from data_io.orientation import SAGITTAL, CORONAL, AXIAL
from skimage.measure import label
from skimage.color import label2rgb
from gui.preferences_viewer import PreferencesManager

LARGE_FONT = ("Verdana", 12)
from dosma import SUPPORTED_SCAN_TYPES, parse_args, SUPPORTED_QUANTITATIVE_VALUES
from msk import knee
from gui.dosma_gui import ScanReader
from gui.gui_utils.filedialog_reader import FileDialogReader
import Pmw


class DosmaViewer(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, DosmaFrame, PageThree, AnalysisFrame):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

        self.pref = PreferencesManager()

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def show_preferences(self):
        self.pref.show_window(self)


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # photo = tk.PhotoImage(file="./defaults/skel-rotate.gif")
        # label1 = tk.Label(image=photo)
        # label1.pack()

        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Scan",
                             command=lambda: controller.show_frame(DosmaFrame))
        button2.pack()

        button3 = ttk.Button(self, text="Knee Analysis",
                             command=lambda: controller.show_frame(AnalysisFrame))
        button3.pack()

        button3 = ttk.Button(self, text="Image Viewer",
                             command=lambda: controller.show_frame(PageThree))
        button3.pack()

        button3 = ttk.Button(self, text="Preferences",
                             command=lambda: controller.show_preferences())
        button3.pack()


class AnalysisFrame(tk.Frame):
    __TISSUES_KEY = 'Tissues'
    __QUANTITATIVE_VALUES_KEY = 'Quantitative values'
    __LOAD_PATH_KEY = 'Load data'

    __PID_KEY = 'pid'
    __MEDIAL_TO_LATERAL_ORIENTATION_KEY = 'ml'

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.manager = dict()
        self.gui_manager = dict()
        self.balloon = Pmw.Balloon()

        self.__init_manager()

        self.__base_gui()
        self.preferences = PreferencesManager()
        self.file_dialog_reader = FileDialogReader()
        self.scan_reader = ScanReader(self)

        button1 = ttk.Button(self, text="Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(anchor='se', side='right')

        button1 = ttk.Button(self, text="Run",
                             command=lambda: self.execute())
        button1.pack(anchor='sw', side='left')

    def execute(self):
        try:
            load_path = self.manager[self.__LOAD_PATH_KEY].get()
            if not load_path:
                raise ValueError('Load path not defined')

            preferences_str = self.preferences.get_cmd_line_str().strip()

            tissue_str = ''
            for c, t in enumerate(self.manager[self.__TISSUES_KEY]):
                if t.get():
                    tissue_str += '--%s ' % knee.SUPPORTED_TISSUES[c].STR_ID
            tissue_str = tissue_str.strip()

            if not tissue_str:
                raise ValueError('No tissues selected')


            qv_str = ''
            for c, qv in enumerate(self.manager[self.__QUANTITATIVE_VALUES_KEY]):
                if qv.get():
                    qv_str += '--%s ' % SUPPORTED_QUANTITATIVE_VALUES[c].name.lower()
            qv_str = qv_str.strip()

            if not qv_str:
                raise ValueError('No quantitative values selected')

            pid = self.manager[self.__PID_KEY].get()
            medial_to_lateral = self.manager[self.__MEDIAL_TO_LATERAL_ORIENTATION_KEY].get()

            if not pid:
                raise ValueError('No PID was provided')

            # analysis string
            str_f = '--l %s %s knee %s --pid %s %s %s' % (load_path,
                                                       preferences_str,
                                                       tissue_str,
                                                       pid,
                                                       '--ml' if medial_to_lateral else '',
                                                          qv_str)
            str_f = str_f.strip()
            parse_args(str_f.split())
        except Exception as e:
            tk.messagebox.showerror(str(type(e)), e.__str__())

    def __init_manager(self):
        self.manager[self.__LOAD_PATH_KEY] = tk.StringVar()
        self.manager[self.__TISSUES_KEY] = [tk.BooleanVar() for i in range(len(knee.SUPPORTED_TISSUES))]
        self.manager[self.__QUANTITATIVE_VALUES_KEY] = [tk.BooleanVar() for i in range(len(SUPPORTED_QUANTITATIVE_VALUES))]

        self.manager[self.__PID_KEY] = tk.StringVar()
        self.manager[self.__MEDIAL_TO_LATERAL_ORIENTATION_KEY] = tk.BooleanVar()

    def __display_pid_info(self):
        hb = tk.Frame(self)
        hb.pack(side='top', anchor='nw')
        l = tk.Label(hb, text=self.__PID_KEY.upper())
        l.pack(side='left', anchor='w', pady=10)
        t = tk.Entry(hb, textvariable=self.manager[self.__PID_KEY])
        t.pack(side='left', anchor='w', pady=10)
        self.balloon.bind(l, 'Patient id')

    def __display_data_loader(self):
        hb = tk.Frame(self)

        filedialog = FileDialogReader(self.manager[self.__LOAD_PATH_KEY])
        b = tk.Button(hb, text=self.__LOAD_PATH_KEY,
                      command=lambda fd=filedialog: self.manager[self.__LOAD_PATH_KEY].set(fd.get_save_dirpath()))
        b.pack(side='left', anchor='nw', pady=10)

        l = tk.Label(hb, textvariable=self.manager[self.__LOAD_PATH_KEY])
        l.pack(side='left', anchor='nw', pady=10)

        hb.pack(side='top', anchor='nw')

    def __display_multi_option(self, label, options_list, boolvar_list):
        hb = tk.Frame(self)
        l = tk.Label(hb, text='%s:' % label)
        l.pack(side='left', anchor='w')
        hb.pack(side='top', anchor='nw')
        frames = [tk.Frame(hb)] * (len(options_list) // 3 + 1)
        for ind, option in enumerate(options_list):
            f = frames[ind // 3]
            b = tk.Checkbutton(f, text=option, variable=boolvar_list[ind])
            b.pack(side='top', anchor='nw', pady=5)

        for f in frames:
            f.pack(side='left', anchor='nw')

        return hb

    def __display_tissues(self):
        tissue_names = [x.FULL_NAME for x in knee.SUPPORTED_TISSUES]
        l = self.__display_multi_option(self.__TISSUES_KEY, tissue_names, self.manager[self.__TISSUES_KEY])
        self.balloon.bind(l, 'Tissues to analyze')

    def __display_quant_vals(self):
        quantitative_value_names = [x.name for x in SUPPORTED_QUANTITATIVE_VALUES]
        l = self.__display_multi_option(self.__QUANTITATIVE_VALUES_KEY, quantitative_value_names,
                                        self.manager[self.__QUANTITATIVE_VALUES_KEY])
        self.balloon.bind(l, 'Quantitative values to analyze')

    def __display_knee_info(self):
        hb = tk.Frame(self)
        hb.pack(side='top', anchor='nw')
        l = tk.Label(hb, text='Medial -> Lateral: ')
        l.pack(side='left', anchor='w', pady=10)
        t = tk.Checkbutton(hb, variable=self.manager[self.__MEDIAL_TO_LATERAL_ORIENTATION_KEY])
        t.pack(side='left', anchor='w', pady=10)

        self.balloon.bind(l, 'Select if Dicoms proceed in medial->lateral direction')

    def __base_gui(self):
        self.__display_data_loader()
        self.__display_pid_info()
        self.__display_tissues()
        self.__display_knee_info()
        self.__display_quant_vals()


class DosmaFrame(tk.Frame):
    __SCAN_KEY = 'Scan'
    __TISSUES_KEY = 'Tissues'

    __DICOM_PATH_KEY = 'Read dicoms'
    __LOAD_PATH_KEY = 'Load data'

    __SAVE_PATH_KEY = 'Save path'

    __DATA_KEY = 'data'  # Track option menu for dicom/load path
    __DATA_PATH_KEY = 'datapath'  # Track filepath associated with option menu

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.file_dialog_reader = FileDialogReader()

        self.manager = dict()
        self.gui_manager = dict()
        self.balloon = Pmw.Balloon()

        self.__init_manager()

        self.__base_gui()
        self.preferences = PreferencesManager()
        self.scan_reader = ScanReader(self)

        button1 = ttk.Button(self, text="Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(anchor='se', side='right')

        button1 = ttk.Button(self, text="Run",
                             command=lambda: self.execute())
        button1.pack(anchor='sw', side='left')

        self.InitUI()

    def execute(self):
        try:
            save_path = self.manager[self.__SAVE_PATH_KEY].get()
            if not save_path:
                raise ValueError('Save path not defined')

            action_str = self.scan_reader.get_cmd_line_str().strip()
            if not action_str:
                raise ValueError('No action selected')

            preferences_str = self.preferences.get_cmd_line_str().strip()

            source = 'd'
            if self.manager[self.__DATA_KEY].get() == self.__LOAD_PATH_KEY:
                source = 'l'

            tissue_str = ''
            for c, t in enumerate(self.manager[self.__TISSUES_KEY]):
                if t.get():
                    tissue_str += '--%s ' % knee.SUPPORTED_TISSUES[c].STR_ID
            tissue_str = tissue_str.strip()

            if not tissue_str:
                raise ValueError('No tissues selected')

            str_f = '--%s %s --s %s %s %s %s %s' % (source, self.manager[self.__DATA_PATH_KEY].get(), save_path,
                                                    preferences_str,
                                                    self.manager[self.__SCAN_KEY].get(),
                                                    tissue_str,
                                                    action_str)

            # print(str_f)

            parse_args(str_f.split())
        except Exception as e:
            tk.messagebox.showerror(str(type(e)), e.__str__())

    def __init_manager(self):
        self.manager[self.__SCAN_KEY] = tk.StringVar()
        self.manager[self.__TISSUES_KEY] = [tk.BooleanVar() for i in range(len(knee.SUPPORTED_TISSUES))]
        self.manager[self.__DATA_KEY] = tk.StringVar()
        self.manager[self.__DATA_PATH_KEY] = tk.StringVar()

        self.manager[self.__SCAN_KEY].trace_add('write', self.__on_scan_change)
        self.manager[self.__SAVE_PATH_KEY] = tk.StringVar()

    def __on_scan_change(self, *args):
        scan_id = self.manager[self.__SCAN_KEY].get()
        scan = None
        for x in SUPPORTED_SCAN_TYPES:
            if x.NAME == scan_id:
                scan = x

        self.scan_reader.load_scan(scan)

        assert scan is not None, "No scan selected"

    def __update_svar(self, *args):
        svar = self.manager[self.__DATA_PATH_KEY]
        selected_option = self.manager[self.__DATA_KEY].get()
        if selected_option == self.__DICOM_PATH_KEY:
            fp = self.file_dialog_reader.get_volume_filepath(selected_option, im_type=fio_utils.ImageDataFormat.dicom)
        elif selected_option == self.__LOAD_PATH_KEY:
            fp = self.file_dialog_reader.get_dirpath(selected_option)
        else:
            raise ValueError('%s key not found' % self.__DATA_KEY)

        if not fp:
            svar.set('')
            return

        svar.set(fp)

        if selected_option == self.__LOAD_PATH_KEY:
            self.manager[self.__SAVE_PATH_KEY].set(fp)

    def __display_data_loader(self):
        s_var = self.manager[self.__DATA_PATH_KEY]

        hb = tk.Frame(self)

        l = tk.Label(hb, text='Data source: ')
        l.pack(side='left', anchor='nw', pady=10)

        options = [self.__DICOM_PATH_KEY, self.__LOAD_PATH_KEY]
        menu = tk.OptionMenu(hb, self.manager[self.__DATA_KEY], *options,
                             command=self.__update_svar)
        menu.pack(side='left', anchor='nw', pady=10)

        l = tk.Label(hb, textvariable=s_var)
        l.pack(side='left', anchor='nw', pady=10)

        hb.pack(side='top', anchor='nw')
        self.balloon.bind(hb, "Read dicoms or load data")

        hb = tk.Frame(self)

        #filedialog = FileDialogReader(self.manager[self.__SAVE_PATH_KEY])
        b = tk.Button(hb, text=self.__SAVE_PATH_KEY,
                      command=lambda fd=self.file_dialog_reader: self.manager[self.__SAVE_PATH_KEY].set(fd.get_save_dirpath()))
        b.pack(side='left', anchor='nw', pady=10)

        l = tk.Label(hb, textvariable=self.manager[self.__SAVE_PATH_KEY])
        l.pack(side='left', anchor='nw', pady=10)

        hb.pack(side='top', anchor='nw')

    def __display_tissues(self):
        hb = tk.Frame(self)
        l = tk.Label(hb, text='Tissues:')
        l.pack(side='left', anchor='w')
        hb.pack(side='top', anchor='nw')
        frames = [tk.Frame(hb)] * (len(knee.SUPPORTED_TISSUES) // 3 + 1)
        for ind, tissue in enumerate(knee.SUPPORTED_TISSUES):
            f = frames[ind // 3]
            b = tk.Checkbutton(f, text=tissue.FULL_NAME, variable=self.manager[self.__TISSUES_KEY][ind])
            b.pack(side='top', anchor='nw', pady=5)

        for f in frames:
            f.pack(side='left', anchor='nw')

        self.balloon.bind(l, 'Tissues to analyze')

        self.balloon.bind(l, 'Select if Dicoms proceed in medial->lateral direction')

    def __base_gui(self):
        self.__display_data_loader()
        self.__display_tissues()

        hb = tk.Frame(self)
        scan_label = tk.Label(hb, text='Scan:')
        scan_label.pack(side='left', anchor='nw', pady=10)
        options = [x.NAME for x in SUPPORTED_SCAN_TYPES]
        scan_dropdown = tk.OptionMenu(hb, self.manager[self.__SCAN_KEY], *options)
        scan_dropdown.pack(side='left', anchor='nw', pady=10)
        hb.pack(side='top', anchor='nw')

    def InitUI(self):
        self.text_box = tk.Text(self, wrap='word', height=11, width=50)
        self.text_box.pack(anchor='s', side='bottom')


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

    def __labeltorgb_3d__(self, im: np.ndarray, labels: np.ndarray, alpha: float = 0.3):
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
