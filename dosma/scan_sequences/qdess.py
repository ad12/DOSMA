"""Quantitative Double Echo in Steady State (qDESS) sequence.

Paper:
    Sveinsson, B., A. S. Chaudhari, G. E. Gold, and B. A. Hargreaves. "A simple analytic method for estimating T2 in the
    knee from DESS." Magnetic resonance imaging 38 (2017): 63-70.
"""
import math
import os
from copy import deepcopy
from typing import Sequence

import numpy as np
from pydicom.tag import Tag

from dosma.scan_sequences.scans import TargetSequence

from dosma.data_io import format_io_utils as fio_utils
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel
from dosma.tissues.tissue import Tissue
from dosma.utils.cmd_line_utils import ActionWrapper
from dosma.quant_vals import T2

import logging

__all__ = ["QDess"]


class QDess(TargetSequence):
    """Handles analysis for qDESS scan sequence.

    qDESS consists of two echos (S1, S2).
    """
    NAME = "qdess"

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

    def __init__(self, dicom_path, load_path=None, **kwargs):
        super().__init__(dicom_path=dicom_path, load_path=load_path, **kwargs)

    def __validate_scan__(self) -> bool:
        """Validate that the dicoms are of qDESS sequence.

        Returns:
            bool: `True` if has 2 echos, `False` otherwise.
        """
        ref_dicom = self.ref_dicom
        # contains_expected_dicom_metadata = self.__GL_AREA_TAG__ in ref_dicom and self.__TG_TAG__ in ref_dicom
        has_expected_num_echos = len(self.volumes) == self.__NUM_ECHOS__

        # return contains_expected_dicom_metadata & has_expected_num_echos
        return has_expected_num_echos

    def segment(self, model: SegModel, tissue: Tissue, use_rms: bool = False):
        """Segment tissue in scan.

        Args:
            model (SegModel): Model to use for segmenting scans.
            tissue (Tissue): The tissue to segment.
            use_rms (`bool`, optional): If `True`, use root-mean-square of
                echos for segmentation (preferred). Defaults to `False`.

        Returns:
            MedicalVolume: Binary mask for segmented region.
        """
        tissue_names = (
            ", ".join([t.FULL_NAME for t in tissue])
            if isinstance(tissue, Sequence) else tissue.FULL_NAME
        )
        logging.info(f"Segmenting {tissue_names}...")

        if use_rms:
            segmentation_volume = self.calc_rms()
        else:
            # Use first echo for segmentation.
            segmentation_volume = self.volumes[0]

        # Segment tissue and add it to list.
        mask = model.generate_mask(segmentation_volume)
        if isinstance(mask, dict):
            if not isinstance(tissue, Sequence):
                tissue = [tissue]
            for abbreviation, tis in zip([t.STR_ID for t in tissue], tissue):
                tis.set_mask(mask[abbreviation])
                self.__add_tissue__(tis)
        else:
            assert isinstance(tissue, Tissue)
            tissue.set_mask(mask)
            self.__add_tissue__(tissue)

        return mask

    def generate_t2_map(self, tissue: Tissue, suppress_fat: bool = False,
                        suppress_fluid: bool = False, beta: float = 1.2,
                        gl_area: float = None, tg: float = None):
        """Generate 3D T2 map.

        Method is detailed in this `paper <https://www.ncbi.nlm.nih.gov/pubmed/28017730>`_.

        Args:
            tissue (Tissue): Tissue to generate T2 map for.
            suppress_fat (`bool`, optional): Suppress fat region in T2 computation. Helps reduce noise.
            suppress_fluid (`bool`, optional): Suppress fluid region in T2 computation. Fluid-nulled image is calculated
                as `S1 - beta*S2`.
            beta (`float`, optional): Beta value used for suppressing fluid. Defaults to 1.2.
            gl_area (`float`, optional): GL Area. Required if not provided in the dicom. Defaults to value in dicom
                tag '0x001910b6'.
            tg: tg value (in microseconds). Required if not provided in the dicom. Defaults to value in dicom tag
                '0x001910b7'.

        Returns:
            qv.T2: T2 fit for tissue.
        """

        if not self.__validate_scan__() and (not gl_area or not tg):
            raise ValueError(
                'dicoms in \'%s\' do not contain GL_Area and Tg tags. Please input manually' % self.dicom_path)

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
        Tg = tg * 1e-6 if tg else float(ref_dicom[self.__TG_TAG__].value) * 1e-6
        T1 = float(tissue.T1_EXPECTED) * 1e-3

        # Flip Angle (degree -> radians)
        alpha = math.radians(float(ref_dicom.FlipAngle))

        GlArea = gl_area if gl_area else float(ref_dicom[self.__GL_AREA_TAG__].value)

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

        # have to divide division into steps to avoid overflow error
        t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

        t2map = np.nan_to_num(t2map)

        # Filter calculated T2 values that are below 0ms and over 100ms
        t2map[t2map <= self.__T2_LOWER_BOUND__] = np.nan
        t2map = np.nan_to_num(t2map)
        t2map[t2map > self.__T2_UPPER_BOUND__] = np.nan
        t2map = np.nan_to_num(t2map)

        t2map = np.around(t2map, self.__T2_DECIMAL_PRECISION__)

        if suppress_fat:
            t2map = t2map * (echo_1 > 0.15 * np.max(echo_1))

        if suppress_fluid:
            vol_null_fluid = echo_1 - beta * echo_2
            t2map = t2map * (vol_null_fluid > 0.1 * np.max(vol_null_fluid))

        t2_map_wrapped = MedicalVolume(t2map,
                                       affine=subvolumes[0].affine,
                                       headers=deepcopy(subvolumes[0].headers))
        t2_map_wrapped = T2(t2_map_wrapped)

        tissue.add_quantitative_value(t2_map_wrapped)

        return t2_map_wrapped

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = preferences.image_data_format):
        """Save data to disk.

        Data will be saved in the directory '`base_save_dirpath`/qdess/'.

        Serializes variables specified in by self.__serializable_variables__().

        Args:
            base_save_dirpath (str): Directory path where all data is stored.
            data_format (ImageDataFormat): Format to save data.
        """
        super().save_data(base_save_dirpath, data_format=data_format)

        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # write echos
        for i in range(len(self.volumes)):
            nii_registration_filepath = os.path.join(base_save_dirpath, 'echo%d.nii.gz' % (i + 1))
            filepath = fio_utils.convert_image_data_format(nii_registration_filepath, data_format)
            self.volumes[i].save_volume(filepath, data_format=data_format)

    def load_data(self, base_load_dirpath):
        """Load data from disk.

        Data will be loaded from the directory '`base_load_dirpath`/qdess'.

        Args:
           base_load_dirpath (str): Directory path where all data is stored.

        Raises:
           NotADirectoryError: if `base_load_dirpath`/qdess/ does not exist.
        """
        super().load_data(base_load_dirpath)

        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        # if reading dicoms from dicom path failed
        if not self.volumes:
            self.volumes = []
            # Load subvolumes from nifti file
            for i in range(self.__NUM_ECHOS__):
                nii_registration_filepath = os.path.join(base_load_dirpath, 'echo%d.nii.gz' % (i + 1))
                subvolume = fio_utils.generic_load(nii_registration_filepath, expected_num_volumes=1)
                self.volumes.append(subvolume)

    def calc_rms(self):
        """Calculate root-mean-square (RMS) of two echos.

        Returns:
            MedicalVolume: Volume with RMS of two echos.
        """
        if self.volumes is None:
            raise ValueError('Volumes must be initialized')

        assert len(self.volumes) == 2, "2 Echos expected"

        echo1 = np.asarray(self.volumes[0].volume, dtype=np.float64)
        echo2 = np.asarray(self.volumes[1].volume, dtype=np.float64)

        assert (~np.iscomplex(echo1)).all() and (~np.iscomplex(echo2)).all()

        sq_sum = echo1 ** 2 + echo2 ** 2

        assert (sq_sum >= 0).all()

        rms = np.sqrt(echo1 ** 2 + echo2 ** 2)

        mv = deepcopy(self.volumes[0])
        mv.volume = rms

        return mv

    @classmethod
    def cmd_line_actions(cls):
        """Provide command line information (such as name, help strings, etc) as list of dictionary."""

        segment_action = ActionWrapper(name=cls.segment.__name__,
                                       help='generate automatic segmentation',
                                       param_help={
                                           'use_rms': 'use root mean square (rms) of two echos for segmentation'},
                                       alternative_param_names={'use_rms': ['rms']})
        generate_t2_map_action = ActionWrapper(name=cls.generate_t2_map.__name__,
                                               aliases=['t2'],
                                               param_help={
                                                   'suppress_fat': 'suppress computation on low SNR fat regions',
                                                   'suppress_fluid': 'suppress computation on fluid regions',
                                                   'beta': 'constant for calculating fluid-nulled image (S1-beta*S2)',
                                                   'gl_area': 'GL Area. Defaults to value in dicom tag \'0x001910b6\'',
                                                   'tg': 'Gradient time (in microseconds). '
                                                         'Defaults to value in dicom tag \'0x001910b7\'.'
                                               },
                                               help='generate T2 map')

        return [(cls.segment, segment_action), (cls.generate_t2_map, generate_t2_map_action)]
