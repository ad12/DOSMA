import matplotlib

matplotlib.use("TkAgg")

from dosma.gui.ims import DosmaViewer  # noqa: E402
from dosma.utils.logger import setup_logger  # noqa: E402

# Initialize logger for the GUI.
setup_logger()

app = DosmaViewer()
app.mainloop()
