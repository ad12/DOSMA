import inspect
import tkinter as tk
from tkinter import IntVar

import Pmw

from dosma import SEGMENTATION_WEIGHTS_DIR_KEY
from gui.gui_utils import gui_utils as gutils
from models.model import SegModel
from tissues.tissue import Tissue


class ScanReader():
    def __init__(self, parent):
        self.parent = parent
        self.hbox = None
        self.action_box = None
        self.params = dict()
        self.action_var = None
        self._action_bool = IntVar()
        self.balloon = None

    def load_scan(self, scan_class):
        self._action_bool = IntVar()
        if self.hbox:
            self.hbox.destroy()
        if self.action_box:
            self.action_box.destroy()
        self.action_var = None
        self.balloon = Pmw.Balloon()

        cmd_line_actions = scan_class.cmd_line_actions()
        hbox = tk.Frame(self.parent)
        hbox.pack(anchor='nw', side='top')

        buttons = []
        count = 0
        for a_method, a_description in cmd_line_actions:
            b = tk.Radiobutton(hbox, text=a_description.name,
                               value=count,
                               command=lambda v=(a_method, a_description): self.show_action_params(v[0], v[1]),
                               variable=self._action_bool)
            self.balloon.bind(b, a_description.help)
            buttons.append(b)
            count += 1

        self._action_bool.set(-1)

        for b in buttons:
            b.pack(anchor='nw', side='left', padx=5)

        self.hbox = hbox

    def show_action_params(self, action, action_wrapper):
        self.action_var = action_wrapper.name
        if self.action_box:
            self.action_box.destroy()

        if self.params:
            self.params = dict()

        hbox = tk.Frame(self.parent)
        hbox.pack(anchor='nw', side='top')
        self.action_box = hbox

        func_signature = inspect.signature(action)
        parameters = func_signature.parameters

        for param_name in parameters.keys():
            param = parameters[param_name]
            param_type = param.annotation
            param_default = param.default

            if param_name == 'self' or param_type is Tissue:
                continue

            # # see if the type is a custom type, if not handle it as a basic type
            is_custom_arg = param_type in CUSTOM_TYPE_TO_GUI
            if is_custom_arg:
                CUSTOM_TYPE_TO_GUI[param_type](self.params, hbox, self.balloon)
                continue

            param_var = gutils.convert_base_type_to_gui(param_name, param_type, param_default, hbox,
                                                        balloon=self.balloon,
                                                        param_help=action_wrapper.get_param_help(param_name))
            self.params[param_name] = (param_var, param_type is not bool)

    def get_cmd_line_str(self):
        if not self.action_var:
            raise ValueError('No action selected. Select an action to continue.')
        cmd_line_str = '%s' % self.action_var
        for param_name in self.params:
            param_var, add_arg = self.params[param_name]

            if add_arg and not param_var.get():
                raise ValueError('\"%s\" must have a value' % param_name)

            if param_var.get():
                cmd_line_str += ' --%s' % param_name
                if add_arg:
                    cmd_line_str += ' %s' % param_var.get()

        return cmd_line_str


def add_segmentation_gui_parser(params, hbox, balloon):
    param_name, param_type, param_default = SEGMENTATION_WEIGHTS_DIR_KEY, str, None
    param_var = gutils.convert_base_type_to_gui(param_name, param_type, param_default, hbox,
                                                balloon=balloon, param_help='path to weights directory')
    params[param_name] = (param_var, param_type is not bool)


CUSTOM_TYPE_TO_GUI = {SegModel: add_segmentation_gui_parser}
