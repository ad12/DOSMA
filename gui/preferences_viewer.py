import tkinter as tk
from tkinter.ttk import Combobox

from tensorflow.python.client import device_lib

from dosma import GPU_KEY, DATA_FORMAT_KEY, VISUALIZATION_FORMAT_KEY

CUDA_DEVICES_STR = "CUDA_VISIBLE_DEVICES"
from data_io.format_io import SUPPORTED_FORMATS, ImageDataFormat
from data_io.fig_format import SUPPORTED_VISUALIZATION_FORMATS

import defaults
import Pmw

LARGE_FONT = ("Verdana", 18)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PreferencesManager(metaclass=Singleton):
    def __init__(self):
        self.frame = None
        self.balloon = None

        self.gui_manager = dict()
        self.gui_elements = dict()

        self.__init_gpu_preferences()
        self.__init_data_format()
        self.__init_visualization_format()

    def __init_gpu_preferences(self):
        gpu_vars = []
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == 'GPU':
                bool_var = tk.BooleanVar()
                x_id = x.name.split(':')[-1]
                gpu_vars.append((x_id, bool_var))

        self.gui_manager['gpu'] = gpu_vars

    def __init_data_format(self):
        int_var = tk.IntVar()
        self.gui_manager['data_format'] = int_var

    def __init_visualization_format(self):
        str_var = tk.StringVar()
        str_var.set(defaults.DEFAULT_FIG_FORMAT)
        self.gui_manager['visualization_format'] = str_var

    @property
    def gpus(self) -> str:
        gpu_ids = []
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == 'GPU':
                x_id = x.name.split(':')[-1]
                gpu_ids.append(x_id)

        if len(gpu_ids) == 0:
            return None

        gpu_str = ''
        for ind, var in enumerate(self.gui_manager['gpu']):
            if not var.get():
                continue
            gpu_str += '%s,' % gpu_ids[ind]

        if len(gpu_str) == 0:
            return None

        gpu_str = gpu_str[:-1]  # remove last comma
        return gpu_str

    @property
    def data_format(self) -> ImageDataFormat:
        return SUPPORTED_FORMATS[self.gui_manager['data_format'].get()]

    @property
    def visualization_format(self) -> str:
        return self.gui_manager['visualization_format'].get()

    def __display_gui(self):
        self.balloon = Pmw.Balloon()

        l = tk.Label(self.frame, text='Preferences', font=LARGE_FONT)
        l.pack()

        hboxes = []
        # show gpu options
        if self.gui_manager['gpu']:
            f = tk.Frame(self.frame)
            gpu_checkboxes = []
            gpu_label = tk.Label(f, text='GPU:')
            self.balloon.bind(gpu_label, 'Select gpus to use for analysis')
            gpu_label.pack()
            for x_id, bool_var in self.gui_manager['gpu']:
                c = tk.Checkbutton(f, text=x_id, variable=bool_var)
                c.pack()
                gpu_checkboxes.append(c)
            hboxes.append(f)

        # show data format options
        f = tk.Frame(self.frame)
        data_format_label = tk.Label(f, text='Data Format:\t')
        self.balloon.bind(data_format_label, 'Select output image data format')
        data_format_label.pack(side='left')
        data_format_var = self.gui_manager['data_format']
        count = 0
        for im_format in SUPPORTED_FORMATS:
            rb = tk.Radiobutton(f, text=im_format.name, variable=data_format_var, value=count)
            rb.pack(side='left')
            count += 1
        hboxes.append(f)

        # show visualization format options
        f = tk.Frame(self.frame)
        visualization_format_label = tk.Label(f, text='Visualization Format:\t')
        self.balloon.bind(visualization_format_label, 'Select figure visualization format')
        visualization_format_label.pack(side='left')
        visualization_format_var = self.gui_manager['visualization_format']
        visualization_format_combobox = Combobox(f,
                                                 state='readonly',
                                                 values=SUPPORTED_VISUALIZATION_FORMATS,
                                                 textvariable=visualization_format_var)

        visualization_format_combobox.pack(side='left')
        hboxes.append(f)

        for f in hboxes:
            f.pack(side='top', anchor='w', pady=10)

    def show_window(self, parent):
        window = tk.Toplevel(parent)
        self.frame = window
        self.__display_gui()

    def get_cmd_line_str(self):
        gpus = self.gpus
        data_format = self.data_format.name
        visualization_format = self.visualization_format
        cmd_line_str = ''
        if gpus:
            cmd_line_str += '--%s %s ' % (GPU_KEY, gpus)
        cmd_line_str += '--%s %s ' % (DATA_FORMAT_KEY, data_format)
        cmd_line_str += '--%s %s ' % (VISUALIZATION_FORMAT_KEY, visualization_format)

        return cmd_line_str.strip()
