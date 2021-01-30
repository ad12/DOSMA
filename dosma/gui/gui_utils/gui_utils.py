# bool --> booltype, default, checkbox
# filepath (str) --> stringtype, default, load file
# str --> stringtype, default, textbox
# float/int -->

import inspect
import tkinter as tk
from tkinter import ttk

from dosma.cli import BASIC_TYPES
from dosma.gui.gui_utils.filedialog_reader import FileDialogReader


class TextWithVar(tk.Text):
    """A text widget that accepts a 'textvariable' option"""

    def __init__(self, parent, *args, **kwargs):
        try:
            self._textvariable = kwargs.pop("textvariable")
        except KeyError:
            self._textvariable = None

        tk.Text.__init__(self, parent, *args, **kwargs)

        # if the variable has data in it, use it to initialize
        # the widget
        if self._textvariable is not None:
            self.insert("1.0", self._textvariable.get())

        # this defines an internal proxy which generates a
        # virtual event whenever text is inserted or deleted
        self.tk.eval(
            """
            proc widget_proxy {widget widget_command args} {

                # call the real tk widget command with the real args
                set result [uplevel [linsert $args 0 $widget_command]]

                # if the contents changed, generate an event we can bind to
                if {([lindex $args 0] in {insert replace delete})} {
                    event generate $widget <<Change>> -when tail
                }
                # return the result from the real widget command
                return $result
            }
            """
        )

        # this replaces the underlying widget with the proxy
        self.tk.eval(
            """
            rename {widget} _{widget}
            interp alias {{}} ::{widget} {{}} widget_proxy {widget} _{widget}
        """.format(
                widget=str(self)
            )
        )

        # set up a binding to update the variable whenever
        # the widget changes
        self.bind("<<Change>>", self._on_widget_change)

        # set up a trace to update the text widget when the
        # variable changes
        if self._textvariable is not None:
            self._textvariable.trace("wu", self._on_var_change)

    def _on_var_change(self, *args):
        """Change the text widget when the associated textvariable changes"""

        # only change the widget if something actually
        # changed, otherwise we'll get into an endless
        # loop
        text_current = self.get("1.0", "end-1c")
        var_current = self._textvariable.get()
        if text_current != var_current:
            self.delete("1.0", "end")
            self.insert("1.0", var_current)

    def _on_widget_change(self, event=None):
        """Change the variable when the widget changes"""
        if self._textvariable is not None:
            self._textvariable.set(self.get("1.0", "end-1c"))


class Filepath(str):
    pass


TYPE_CAST = {bool: tk.BooleanVar, str: tk.StringVar, int: tk.IntVar, float: tk.DoubleVar}


def contains_filepath_keywords(param_name: str):
    fp_keywords = ["dir", "path", "directory", "file"]
    for k in fp_keywords:
        if k in param_name:
            return True

    return False


def convert_base_type_to_gui(param_name, param_type, param_default, root, **kwargs):
    balloon = None
    param_help = ""
    if "balloon" in kwargs:
        balloon = kwargs.get("balloon")
    if "param_help" in kwargs:
        param_help = kwargs.get("param_help")

    assert param_type in BASIC_TYPES, "type %s not in BASIC_TYPES" % param_type

    # add default value to param help
    has_default = param_default is not inspect._empty and param_default is not None

    type_var = TYPE_CAST[param_type]()
    if has_default:
        type_var.set(param_default)

    is_filepath = (
        (param_type is str) and (not has_default) and contains_filepath_keywords(param_name)
    )
    hbox = None
    if is_filepath:
        hbox = format_filepath_gui(root, param_name, type_var)
    elif param_type is bool:
        hbox = format_bool_gui(root, param_name, type_var)
    elif param_type is str:
        if "options" in kwargs:
            hbox = format_list_gui(root, param_name, type_var, **kwargs)
        else:
            hbox = format_str_gui(root, param_name, type_var)

    # TODO: Add suport for float and int values.
    if hbox:
        if balloon and param_help:
            balloon.bind(hbox, param_help)

    return type_var


def format_filepath_gui(root, label, type_var, **kwargs):
    hbox = tk.Frame(root)
    hbox.pack(side="top", anchor="nw")

    _label = tk.Label(hbox, text="%s: " % label)
    _label.pack(side="left", anchor="nw", padx=5)

    t = tk.Label(hbox, textvariable=type_var)
    t.pack(side="left", anchor="nw", padx=5)

    fd = FileDialogReader(type_var)
    fd_prompt = "Load %s" % label.lower()

    f_action = fd.get_filepath
    if "dir" in label.lower():
        f_action = fd.get_dirpath

    b = ttk.Button(root, text=fd_prompt, command=lambda: f_action(title=fd_prompt))

    b.pack(anchor="nw", pady=1)

    return hbox


def format_str_gui(root, label, type_var, **kwargs):
    hbox = tk.Frame(root)
    hbox.pack(side="top", anchor="nw")

    _label = tk.Label(hbox, text="%s: " % label)
    _label.pack(side="left", anchor="nw", padx=5)

    t = TextWithVar(hbox, textvariable=type_var)
    t.pack(side="left", anchor="nw", padx=5)

    return hbox


def format_bool_gui(root, label, type_var, **kwargs):
    hbox = tk.Frame(root)
    hbox.pack(side="top", anchor="nw")

    _label = tk.Label(hbox, text="%s: " % label)
    _label.pack(side="left", anchor="nw", padx=5)

    t = tk.Checkbutton(hbox, variable=type_var)
    t.pack(side="left", anchor="nw", padx=5)

    return hbox


def format_list_gui(root, label, type_var, **kwargs):
    options = kwargs.get("options")

    hbox = tk.Frame(root)
    hbox.pack(side="top", anchor="nw")

    _label = tk.Label(hbox, text="%s: " % label)
    _label.pack(side="left", anchor="nw", padx=5)

    t = tk.OptionMenu(hbox, type_var, *options)
    t.pack(side="left", anchor="nw", padx=5)

    return hbox
