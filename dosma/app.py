import matplotlib

matplotlib.use("TkAgg")

from dosma.file_constants import LOG_FILE_PATH  # noqa: E402
from dosma.gui.ims import DosmaViewer  # noqa: E402
from dosma.utils import io_utils  # noqa: E402

# Initialize logger for the GUI.
io_utils.init_logger(LOG_FILE_PATH)

app = DosmaViewer()
app.mainloop()
