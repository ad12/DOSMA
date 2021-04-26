import matplotlib

matplotlib.use("TkAgg")

from dosma.gui.ims import DosmaViewer  # noqa: E402
from dosma.utils import env  # noqa: E402
from dosma.utils import io_utils  # noqa: E402

# Initialize logger for the GUI.
io_utils.init_logger(env.log_file_path())

app = DosmaViewer()
app.mainloop()
