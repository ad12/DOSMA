import matplotlib
matplotlib.use("TkAgg")

from dosma.file_constants import LOG_FILE_PATH
from dosma.gui.ims import DosmaViewer
from dosma.utils import io_utils

# Initialize logger for the GUI.
io_utils.init_logger(LOG_FILE_PATH)

app = DosmaViewer()
app.mainloop()
