import math
import os
from copy import deepcopy

import numpy as np
from pydicom.tag import Tag

from data_io import format_io_utils as fio_utils
from data_io.med_volume import MedicalVolume
from data_io.nifti_io import NiftiReader
from scan_sequences.scans import TargetSequence
from utils.quant_vals import T2


class Dess(TargetSequence):
    """Handles analysis for DESS scan sequence """
    NAME = 'dess'

    # DESS DICOM header keys
    __GL_AREA_TAG__ = Tag(0x001910b6)
    __TG_TAG__ = Tag(0x001910b7)

    # DESS constants
    __NUM_ECHOS__ = 2
    __VOLUME_DIMENSIONS__ = 3
    __D__ = 1.25 * 1e-9

    # Clipping bounds for t2
    __T2_LOWER_BOUND__ = 0
    __T2_UPPER_BOUND__ = 100
    __T2_DECIMAL_PRECISION__ = 1  # 0.1 ms

    use_rms = False

    def __init__(self, dicom_path, dicom_ext=None, load_path=None):
        super().__init__(dicom_path, dicom_ext, load_path)

        if not self.validate_dess():
            raise ValueError('dicoms in \'%s\' are not acquired from DESS sequence' % self.dicom_path)

    def validate_dess(self):
        """Validate that the dicoms are of DESS sequence by checking for dicom header tags
        :return: a boolean
        """
        ref_dicom = self.ref_dicom
        return self.__GL_AREA_TAG__ in ref_dicom and self.__TG_TAG__ in ref_dicom and len(
            self.volumes) == self.__NUM_ECHOS__

    def segment(self, model, tissue):
        # Use first echo for segmentation
        print('Segmenting %s...' % tissue.FULL_NAME)

        if self.use_rms:
            segmentation_volume = self.calc_rms()
        else:
            segmentation_volume = self.volumes[0]

        # Segment tissue and add it to list
        mask = model.generate_mask(segmentation_volume)
        tissue.set_mask(mask)

        self.__add_tissue__(tissue)

        return mask

    def generate_t2_map(self, tissue):
        """ Generate 3D t2 map

        :return MedicalVolume with 3D map of t2 values
                all invalid pixels are denoted by the value 0
        """

        if self.volumes is None or self.ref_dicom is None:
            raise ValueError('volumes and ref_dicom fields must be initialized')

        ref_dicom = self.ref_dicom

        r, c, num_slices = self.volumes[0].volume.shape
        subvolumes = self.volumes

        # Split echos
        echo_1 = subvolumes[0].volume
        echo_2 = subvolumes[1].volume

        # All timing in seconds
        TR = float(ref_dicom.RepetitionTime) * 1e-3
        TE = float(ref_dicom.EchoTime) * 1e-3
        Tg = float(ref_dicom[self.__TG_TAG__].value) * 1e-6
        T1 = float(tissue.T1_EXPECTED) * 1e-3

        # Flip Angle (degree -> radians)
        alpha = math.radians(float(ref_dicom.FlipAngle))

        GlArea = float(ref_dicom[self.__GL_AREA_TAG__].value)

        Gl = GlArea / (Tg * 1e6) * 100
        gamma = 4258 * 2 * math.pi  # Gamma, Rad / (G * s).
        dkL = gamma * Gl * Tg

        # Simply math
        k = math.pow((math.sin(alpha / 2)), 2) * (
                1 + math.exp(-TR / T1 - TR * math.pow(dkL, 2) * self.__D__)) / (
                    1 - math.cos(alpha) * math.exp(-TR / T1 - TR * math.pow(dkL, 2) * self.__D__))

        c1 = (TR - Tg / 3) * (math.pow(dkL, 2)) * self.__D__

        # T2 fit
        mask = np.ones([r, c, num_slices])

        ratio = mask * echo_2 / echo_1
        ratio = np.nan_to_num(ratio)

        t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

        t2map = np.nan_to_num(t2map)

        # Filter calculated T2 values that are below 0ms and over 100ms
        t2map[t2map <= self.__T2_LOWER_BOUND__] = np.nan
        t2map = np.nan_to_num(t2map)
        t2map[t2map > self.__T2_UPPER_BOUND__] = np.nan
        t2map = np.nan_to_num(t2map)

        t2map = np.around(t2map, self.__T2_DECIMAL_PRECISION__)

        t2_map_wrapped = MedicalVolume(t2map, subvolumes[0].pixel_spacing, subvolumes[0].orientation,
                                       subvolumes[0].scanner_origin)

        tissue.add_quantitative_value(T2(t2_map_wrapped))

        return t2map

    def save_data(self, base_save_dirpath, data_format='nifti'):
        super().save_data(base_save_dirpath)

        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # write echos
        for i in range(len(self.volumes)):
            nii_registration_filepath = os.path.join(base_save_dirpath, 'echo%d.nii.gz' % (i + 1))
            filepath = fio_utils.convert_format_filename(nii_registration_filepath, data_format)
            self.volumes[i].save_volume(filepath, data_format=data_format)

    def load_data(self, base_load_dirpath):
        super().load_data(base_load_dirpath)

        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        self.volumes = []
        # Load subvolumes from nifti file
        for i in range(self.__NUM_ECHOS__):
            nii_registration_filepath = os.path.join(base_load_dirpath, 'echo%d.nii.gz' % (i + 1))
            subvolume = NiftiReader().load(nii_registration_filepath)
            self.volumes.append(subvolume)

    def calc_rms(self):
        """Calculate RMS of 2 echos
        :return: A MedicalVolume
        """
        if self.volumes is None:
            raise ValueError('Volumes must be initialized')

        assert len(self.volumes) == 2, "2 Echos expected"

        echo1 = np.asarray(self.volumes[0].volume, dtype=np.float64)
        echo2 = np.asarray(self.volumes[1].volume, dtype=np.float64)

        assert (echo1 >= 0).all()
        assert (echo2 >= 0).all()

        assert (~np.iscomplex(echo1)).all() and (~np.iscomplex(echo2)).all()

        sq_sum = echo1 ** 2 + echo2 ** 2

        assert (sq_sum >= 0).all()

        rms = np.sqrt(echo1 ** 2 + echo2 ** 2)

        mv = deepcopy(self.volumes[0])
        mv.volume = rms

        return mv
