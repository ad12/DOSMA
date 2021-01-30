import logging
import sys
import tkinter as tk


class WidgetLogger(logging.StreamHandler):
    def __init__(self, widget):
        logging.Handler.__init__(self, sys.stdout)
        self.setLevel(logging.INFO)
        self.widget = widget
        self.widget.config(state="disabled")

    def emit(self, record):
        self.widget.config(state="normal")
        # Append message (record) to the widget
        self.widget.insert(tk.END, self.format(record) + "\n")
        self.widget.see(tk.END)  # Scroll to the bottom
        self.widget.config(state="disabled")
