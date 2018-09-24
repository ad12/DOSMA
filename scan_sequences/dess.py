import math, os
import numpy as np
from pydicom.tag import Tag

from scan_sequences.scans import TargetSequence
from utils import dicom_utils, im_utils, io_utils
from utils.quant_vals import QuantitativeValue


class Dess(TargetSequence):
    NAME = 'dess'

    # DESS DICOM header keys
    __GL_AREA_TAG__ = Tag(0x001910b6)
    __TG_TAG__ = Tag(0x001910b7)

    # DESS constants
    __NUM_ECHOS__ = 2
    __VOLUME_DIMENSIONS__ = 3
    __T1__ = 1.2
    __D__ = 1.25 * 1e-9

    # Clipping bounds for t2
    __T2_LOWER_BOUND__ = 0.1
    __T2_UPPER_BOUND__ = 100
    __T2_DECIMAL_PRECISION__ = 1 # 0.1 ms

    def __init__(self, dicom_path, dicom_ext=None):
        super().__init__(dicom_path, dicom_ext)
        self.ref_dicom = self.refs_dicom[0]
        self.subvolumes = self.split_volume()
        if not self.validate_dess():
            raise ValueError('dicoms in \'%s\' are not acquired from DESS sequence' % self.dicom_path)

    def split_volume(self):
        volume = self.volume
        echos = self.__NUM_ECHOS__

        if len(volume.shape) != self.__VOLUME_DIMENSIONS__:
            raise ValueError(
                "Dimension Error: input has %d dimensions. Expected %d" % (volume.ndims, self.__VOLUME_DIMENSIONS__))
        if echos <= 0:
            raise ValueError('There must be at least 1 echo per volume')

        depth = volume.shape[2]
        if depth % echos != 0:
            raise ValueError('Number of slices per echo must be the same')

        sub_volumes = []
        for i in range(echos):
            sub_volumes.append(volume[:, :, i::echos])

        return sub_volumes

    def validate_dess(self):
        """
        Validate that the dicoms are actually dess
        :return:
        """
        ref_dicom = self.ref_dicom
        return self.__GL_AREA_TAG__ in ref_dicom and self.__TG_TAG__ in ref_dicom

    def segment(self, model, tissue):
        # Use first echo for segmentation
        segmentation_volume = self.subvolumes[0]
        volume = dicom_utils.whiten_volume(segmentation_volume)

        # Segment tissue and add it to list
        mask = model.generate_mask(volume)
        tissue.mask = mask
        self.__add_tissue__(tissue)

        return mask

    def save_tissue_masks(self, dirpath, ext='tiff'):
        for tissue in self.tissues:
            filepath = os.path.join(dirpath, '%s.%s' % (tissue.NAME, ext))
            im_utils.write_3d(filepath, tissue.mask)

    def generate_t2_map(self):
        """ Generate t2 map
        :param dicom_array: 3D numpy array in dual echo format
                            (echo 1 = dicom_array[:,:,0::2], echo 2 = dicom_array[:,:,1::2])
        :param ref_dicom: a pydicom reference/header

        :rtype: 2D numpy array with values (0, 100]
                all voxel values of magnitude outside of this range are invalid
                all invalid pixels are denoted by the value 0
        """

        dicom_array = self.volume
        ref_dicom = self.ref_dicom

        if len(dicom_array.shape) != 3:
            raise ValueError("dicom_array must be 3D volume")

        r, c, num_slices = dicom_array.shape
        subvolumes = self.subvolumes

        # Split echos
        echo_1 = subvolumes[0]
        echo_2 = subvolumes[1]

        # All timing in seconds
        TR = float(ref_dicom.RepetitionTime) * 1e-3
        TE = float(ref_dicom.EchoTime) * 1e-3
        Tg = float(ref_dicom[self.__TG_TAG__].value) * 1e-6

        # Flip Angle (degree -> radians)
        alpha = math.radians(float(ref_dicom.FlipAngle))

        GlArea = float(ref_dicom[self.__GL_AREA_TAG__].value)

        Gl = GlArea / (Tg * 1e6) * 100
        gamma = 4258 * 2 * math.pi  # Gamma, Rad / (G * s).
        dkL = gamma * Gl * Tg

        # Simply math
        k = math.pow((math.sin(alpha / 2)), 2) * (1 + math.exp(-TR / self.__T1__ - TR * math.pow(dkL, 2) * self.__D__)) / (
                    1 - math.cos(alpha) * math.exp(-TR / self.__T1__ - TR * math.pow(dkL, 2) * self.__D__))

        c1 = (TR - Tg / 3) * (math.pow(dkL, 2)) * self.__D__

        # T2 fit
        mask = np.ones([r, c, int(num_slices / 2)])

        ratio = mask * echo_2 / echo_1
        ratio = np.nan_to_num(ratio)

        t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

        t2map = np.nan_to_num(t2map)

        # Filter calculated T2 values that are below 0ms and over 100ms
        t2map[t2map < self.__T2_LOWER_BOUND__] = 0.0
        t2map[t2map > self.__T2_UPPER_BOUND__] = 0.0
        t2map[np.isnan(t2map)] = 0.0
        t2map[np.isinf(t2map)] = 0.0

        t2map = np.around(t2map, self.__T2_DECIMAL_PRECISION__)

        self.t2map = t2map

        return t2map

    def save_data(self, save_dirpath):
        data = {QuantitativeValue.T2.name: self.t2map}
        io_utils.save_h5(os.path.join(save_dirpath, self.__data_filename__()), data)

    def load_data(self, load_dirpath):
        data = io_utils.load_h5(os.path.join(load_dirpath, self.__data_filename__()))
        self.t2map = data[QuantitativeValue.T2.name]