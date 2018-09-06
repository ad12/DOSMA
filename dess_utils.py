import numpy as np
import math
from pydicom.tag import Tag

import utils


NUM_ECHOS = 2


# DESS DICOM header keys
DESS_GL_AREA_TAG = Tag(0x001910b6)
DESS_TG_TAG = Tag(0x001910b7)

T1 = 1.2
D = 1.25*1e-9


def calc_t2_map(dicom_array, ref_dicom):
    if len(dicom_array.shape) != 3:
        raise ValueError("dicom_array must be 3D volume")

    r, c, num_slices = dicom_array.shape

    # Split echos
    subvolumes = utils.split_volume(dicom_array, 2)
    echo_1 = subvolumes[0]
    echo_2 = subvolumes[1]

    # All timing in seconds
    TR = float(ref_dicom.RepetitionTime) * 1e-3
    TE = float(ref_dicom.EchoTime) * 1e-3
    Tg = float(ref_dicom[DESS_TG_TAG].value) * 1e-6

    # Flip Angle (degree -> radians)
    alpha = math.radians(float(ref_dicom.FlipAngle))

    GlArea = float(ref_dicom[DESS_GL_AREA_TAG].value)

    Gl = GlArea / (Tg * 1e6) * 100
    gamma = 4258 * 2 * math.pi # Gamma, Rad / (G * s).
    dkL = gamma * Gl * Tg

    # Simply math
    k = math.pow((math.sin(alpha / 2)), 2) * (1 + math.exp(-TR/T1 - TR * math.pow(dkL, 2) * D)) / (1 - math.cos(alpha)* math.exp(-TR/T1 - TR * math.pow(dkL, 2) * D))

    c1 = (TR - Tg/3) * (math.pow(dkL, 2)) * D

    # T2 fit
    mask = np.ones([r, c, int(num_slices / 2)])

    ratio = mask * echo_2 / echo_1
    ratio = np.nan_to_num(ratio)

    t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

    t2map = np.nan_to_num(t2map)

    # Filter calculated T2 values that are below 0ms and over 100ms
    t2map[t2map <= 0] = np.nan
    t2map[t2map > 100] = np.nan

    return t2map


def get_tissue_t2(t2_map, mask, tissue):
    # TODO: unroll masks, multiply with t2_map, and get tissue values
    return None

