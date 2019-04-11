import os
from tkinter import StringVar
from tkinter import filedialog

from data_io import format_io_utils as fio_utils
from gui.preferences_viewer import PreferencesManager


class FileDialogReader():
    SUPPORTED_FORMATS = (('nifti files', '*.nii\.gz'), ('dicom files', '*.dcm'))
    __base_filepath = '../'

    def __init__(self, string_var: StringVar = None):
        self.preferences = PreferencesManager()
        self.string_var = string_var

    def load_volume(self, title='Select volume file(s)'):
        filepath = self.get_volume_filepath(title)

        im = fio_utils.generic_load(filepath, 1)

        return im

    def get_volume_filepath(self, title='Select path', im_type: fio_utils.ImageDataFormat = None):
        filetypes = None
        if im_type is fio_utils.ImageDataFormat.dicom:
            filetypes = ((im_type.name, "*.dcm"),)

        files = filedialog.askopenfilenames(initialdir=self.__base_filepath, title=title, filetypes=filetypes)
        if len(files) == 0:
            return

        filepath = files[0]
        self.__base_filepath = os.path.dirname(filepath)

        if filepath.endswith('.dcm'):
            filepath = os.path.dirname(filepath)

        if self.string_var:
            self.string_var.set(filepath)

        return filepath

    def get_filepath(self, title='Select file'):
        file_str = filedialog.askopenfilename(initialdir=self.__base_filepath, title=title)
        if not file_str:
            return

        if self.string_var:
            self.string_var.set(file_str)

        return file_str

    def get_dirpath(self, title='Select directory'):
        file_str = filedialog.askdirectory(initialdir=self.__base_filepath, title=title)
        if not file_str:
            return

        if self.string_var:
            self.string_var.set(file_str)

        return file_str

    def get_save_dirpath(self):
        file_str = filedialog.askdirectory(initialdir=self.__base_filepath, mustexist=False)
        if not file_str:
            return

        if self.string_var:
            self.string_var.set(file_str)

        return file_str
