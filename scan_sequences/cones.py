import os

from scan_sequences.scans import NonTargetSequence
from utils import dicom_utils, io_utils
from utils import quant_vals as qv
from nipype.interfaces.elastix import Registration, ApplyWarp
import numpy as np
import file_constants as fc
import scipy.ndimage as sni
from natsort import natsorted

__EXPECTED_NUM_ECHO_TIMES__ = 4


class Cones(NonTargetSequence):
    NAME = 'cones'


if __name__ == '__main__':
    cq = Cones('../dicoms/healthy07/009', 'dcm', './')