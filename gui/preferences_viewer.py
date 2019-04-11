import tkinter as tk

from tensorflow.python.client import device_lib

from dosma import GPU_KEY, DATA_FORMAT_KEY

CUDA_DEVICES_STR = "CUDA_VISIBLE_DEVICES"
from data_io.format_io import SUPPORTED_FORMATS, ImageDataFormat

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
        self.gui_manager = dict()
        self.gui_elements = dict()

        self.__init_gpu_preferences()
        self.__init_data_format()

    def __init_gpu_preferences(self):
        gpu_vars = []
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == 'GPU':
                bool_var = tk.BooleanVar()
                x_id = x.name.split(':')[-1]
                gpu_vars.append((x_id, bool_var))

        self.gui_manager['gpu'] = gpu_vars

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

    def __init_data_format(self):
        int_var = tk.IntVar()
        self.gui_manager['data_format'] = int_var

    @property
    def data_format(self) -> ImageDataFormat:
        return SUPPORTED_FORMATS[self.gui_manager['data_format'].get()]

    def __display_gui(self):
        l = tk.Label(self.frame, text='Preferences', font=LARGE_FONT)
        l.pack()

        # show gpu options
        if self.gui_manager['gpu']:
            gpu_checkboxes = []
            gpu_label = tk.Label(self.frame, text='GPU:')
            gpu_label.pack()
            for x_id, bool_var in self.gui_manager['gpu']:
                c = tk.Checkbutton(self.frame, text=x_id, variable=bool_var)
                c.pack()
                gpu_checkboxes.append(c)

        # show data format options
        data_format_label = tk.Label(self.frame, text='Data Format:\t')
        data_format_label.pack(side='left')
        data_format_var = self.gui_manager['data_format']
        count = 0
        for im_format in SUPPORTED_FORMATS:
            rb = tk.Radiobutton(self.frame, text=im_format.name, variable=data_format_var, value=count)
            rb.pack(side='left')
            count += 1

    def show_window(self, parent):
        window = tk.Toplevel(parent)
        self.frame = window
        self.__display_gui()

    def get_cmd_line_str(self):
        gpus = self.gpus
        data_format = self.data_format.name
        cmd_line_str = ''
        if gpus:
            cmd_line_str += '--%s %s' % (GPU_KEY, gpus)
        cmd_line_str += '--%s %s' % (DATA_FORMAT_KEY, data_format)

        return cmd_line_str
