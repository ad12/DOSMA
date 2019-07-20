import matplotlib
matplotlib.use("TkAgg")

from dosma.gui.ims import DosmaViewer

app = DosmaViewer()
app.mainloop()
