"""Default parameters
Please change depending on your preferences

This is the first file that is imported from this software - all initialization imports should occur here
"""

# Parse preferences file

# Default inference batch size - based on RAM of computer
# for 16 GB RAM , recommend 16
DEFAULT_BATCH_SIZE = 16

# Fix the bounds for the visualization colorbar based on the quantitative measure (T2, T1rho, etc)
# soft bound: if any value exceeds these bounds, automatically set this parameter to False
# hard bound: clip all values out of bounds, so that the color bar is fixed (good for comparing images)
VISUALIZATION_SOFT_BOUNDS = False
VISUALIZATION_HARD_BOUNDS = True

# The dilation rate we use for dilating any mask before registration
DEFAULT_MASK_DIL_RATE = 4.0
DEFAULT_MASK_DIL_THRESHOLD = 0.1

# The R^2 fit threshold to include when estimating quantitative values
DEFAULT_R2_THRESHOLD = 0.9

# DPI to save images
DEFAULT_DPI = 400

# Matplotlib initialization
import matplotlib

matplotlib.use('Agg')

# Default font size
DEFAULT_FONT_SIZE = 16
DEFAULT_TEXT_SPACING = DEFAULT_FONT_SIZE * 0.01

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': DEFAULT_FONT_SIZE})
