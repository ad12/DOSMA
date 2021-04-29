import tkinter as tk
from tkinter import ttk
from typing import Dict

import Pmw

from dosma.cli import GPU_KEY
from dosma.core.io.format_io import ImageDataFormat
from dosma.defaults import preferences
from dosma.utils import env

if env.package_available("tensorflow"):
    from tensorflow.python.client import device_lib
else:
    device_lib = None


CUDA_DEVICES_STR = "CUDA_VISIBLE_DEVICES"
SUPPORTED_IMAGE_DATA_FORMATS = list(ImageDataFormat)
LARGE_FONT = ("Verdana", 18)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


TYPE_CAST = {bool: tk.BooleanVar, str: tk.StringVar, int: tk.IntVar, float: tk.DoubleVar}


class CommandLineFlagGUI:
    def __init__(self, cmd_line_metadata: dict, **kwargs):
        self._cmd_line_metadata = cmd_line_metadata
        self._name = cmd_line_metadata["name"]
        self._default = cmd_line_metadata["default"]
        self._help = cmd_line_metadata["help"]
        self._type = cmd_line_metadata["type"]
        self._choices = (
            cmd_line_metadata["choices"] if "choices" in cmd_line_metadata.keys() else None
        )

        # Tkinter Variable used for binding with GUI.
        self._tk_var = TYPE_CAST[self._type]()
        self._tk_var.set(self._default)

        # GUI formatting.
        self._label_format = kwargs.get("label_format") if "label_format" in kwargs else "%s: "

        # Memory
        self._widget = None

    def draw(self, frame: tk.Frame, **kwargs):
        """Draw element best suited for command line metadata.
        :param frame: The frame where to add GUI element
        :type frame: A Tkinter frame
        """
        draw_cmd = {
            str: lambda root: self._draw_str_gui(root),
            bool: lambda root: self._draw_bool_gui(root),
            int: lambda root: self._draw_number_gui(root, "float"),
            float: lambda root: self._draw_number_gui(root, "float"),
        }

        padx = kwargs.get("padx") if "padx" in kwargs else 5
        balloon = kwargs.get("balloon") if "balloon" in kwargs else None

        hbox = tk.Frame(frame)
        hbox.pack(side="top", anchor="nw")

        label = tk.Label(hbox, text=self._label_format % self._name)
        label.pack(side="left", anchor="nw", padx=padx)

        if self._choices:
            t = self._draw_list_gui(hbox)
        else:
            t = draw_cmd[self._type](hbox)

        t.pack(side="left", anchor="nw", padx=padx)
        self._widget = t

        if balloon:
            balloon.bind(label, self._help)

        return hbox

    def _draw_str_gui(self, root):
        return tk.Entry(root, textvariable=self.tk_var)

    def _draw_bool_gui(self, root):
        return tk.Checkbutton(root, variable=self.tk_var)

    def _draw_number_gui(self, root, dtype="float"):
        vcmd = (
            root.register(self._validate_number),
            dtype,
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )
        return tk.Entry(root, textvariable=self.tk_var, validate="all", validatecommand=vcmd)

    def _draw_int_gui(self, root):
        vcmd = (
            root.register(self._validate_number),
            "float",
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )
        return tk.Entry(root, textvariable=self.tk_var, validate="all", validatecommand=vcmd)

    def _validate_number(
        self,
        dtype,
        action,
        index,
        value_if_allowed,
        prior_value,
        text,
        validation_type,
        trigger_type,
        widget_name,
    ):
        if trigger_type not in ["key", "focusout"]:
            return True

        if not value_if_allowed:
            if trigger_type == "key":
                return True
            else:
                self.tk_var.set(self._default)
                self._widget["validate"] = validation_type
                return False

        try:
            eval(dtype)(value_if_allowed)
            return True
        except ValueError:
            return False

    def _draw_list_gui(self, root):
        options = self._choices
        return tk.OptionMenu(root, self.tk_var, *options)

    @property
    def tk_var(self):
        return self._tk_var


class PreferencesManager(metaclass=Singleton):
    def __init__(self):
        self.frame = None
        self.balloon = None

        self.gui_manager: Dict = {}
        self.gui_elements: Dict = {}

        self._init_gpu_preferences()

        # Init preferences
        self._preference_elements = {}
        preferences_metadata = preferences.cmd_line_flags()
        for preference in preferences_metadata.keys():
            self._preference_elements[preference] = CommandLineFlagGUI(
                preferences_metadata[preference], label_format="%s:\t"
            )

    def _init_gpu_preferences(self):
        gpu_vars = []
        if device_lib is not None:
            local_device_protos = device_lib.list_local_devices()
            for x in local_device_protos:
                if x.device_type == "GPU":
                    bool_var = tk.BooleanVar()
                    x_id = x.name.split(":")[-1]
                    gpu_vars.append((x_id, bool_var))

        self.gui_manager["gpu"] = gpu_vars

    @property
    def gpus(self) -> str:
        if device_lib is None:
            return None

        gpu_ids = []
        local_device_protos = device_lib.list_local_devices()
        for x in local_device_protos:
            if x.device_type == "GPU":
                x_id = x.name.split(":")[-1]
                gpu_ids.append(x_id)

        if len(gpu_ids) == 0:
            return None

        gpu_str = ""
        for ind, var in enumerate(self.gui_manager["gpu"]):
            if not var.get():
                continue
            gpu_str += "%s," % gpu_ids[ind]

        if len(gpu_str) == 0:
            return None

        gpu_str = gpu_str[:-1]  # remove last comma
        return gpu_str

    def _restore_preference_default(self):
        for preference in self._preference_elements.keys():
            self._preference_elements[preference].tk_var.set(preferences.get(preference))

    def __display_gui(self):
        self._restore_preference_default()
        self.balloon = Pmw.Balloon()

        _label = tk.Label(self.frame, text="Preferences", font=LARGE_FONT)
        _label.pack()

        hboxes = []
        # show gpu options
        if self.gui_manager["gpu"]:
            f = tk.Frame(self.frame)
            gpu_checkboxes = []
            gpu_label = tk.Label(f, text="GPU:")
            self.balloon.bind(gpu_label, "Select gpus to use for analysis")
            gpu_label.pack()
            for x_id, bool_var in self.gui_manager["gpu"]:
                c = tk.Checkbutton(f, text=x_id, variable=bool_var)
                c.pack()
                gpu_checkboxes.append(c)
            hboxes.append(f)

        # Add command line governed preferences
        for preference in self._preference_elements.keys():
            hbox = self._preference_elements[preference].draw(self.frame, balloon=self.balloon)
            hboxes.append(hbox)

        for f in hboxes:
            f.pack(side="top", anchor="w", pady=10)

        # Add apply settings and save preferences buttons
        # apply settings: save preferences for this session only
        # save settings: changes preferences in file
        hbox = tk.Frame(self.frame)
        apply_settings_button = ttk.Button(
            hbox, text="Apply Settings", command=lambda: self._apply_settings()
        )
        self.balloon.bind(apply_settings_button, "Apply settings for this session")
        save_settings_button = ttk.Button(
            hbox, text="Save Settings", command=lambda: self._save_settings()
        )
        self.balloon.bind(save_settings_button, "Save settings for all future sessions")
        apply_settings_button.pack(side="left", anchor="nw", padx=5)
        save_settings_button.pack(side="left", anchor="nw", padx=5)
        hbox.pack(side="bottom", anchor="se")

    def _apply_settings(self):
        for preference in self._preference_elements.keys():
            new_val = self._preference_elements[preference].tk_var.get()
            preferences.set(preference, new_val)

    def _save_settings(self):
        self._apply_settings()
        preferences.save()

    def show_window(self, parent):
        window = tk.Toplevel(parent)
        self.frame = window
        self.__display_gui()

    def get_cmd_line_str(self):
        gpus = self.gpus
        cmd_line_str = ""
        if gpus:
            cmd_line_str += "--%s %s " % (GPU_KEY, gpus)

        return cmd_line_str.strip()
