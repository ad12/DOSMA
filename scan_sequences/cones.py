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

    def __init__(self, dicom_path=None, dicom_ext=None, save_dir=None, interregistered_volumes_path=''):
        super().__init__(dicom_path, dicom_ext)
        self.subvolumes = None

        if dicom_path is not None:
            self.save_dir = save_dir

            self.subvolumes = self.__split_volumes__(__EXPECTED_NUM_ECHO_TIMES__)
            self.intermediate_save_dir = os.path.join(self.save_dir, self.NAME)
        elif interregistered_volumes_path:
            self.subvolumes = self.__load_interregistered_files__(interregistered_volumes_path)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or interregistered_volumes_path must be specified')

    def interregister(self, target, mask=None):
        subvolumes = self.subvolumes

        echo_times = natsorted(list(subvolumes.keys()))
        base_echo_time = echo_times[-1]
        base_subvolume_filepath = ''

        file_data = dict()
        transformation_echo_times = []

        # Save all subvolumes to temp folder
        for echo_time in echo_times:
            filepath = os.path.join(self.temp_path, '%03d.nii' % echo_time)
            io_utils.save_nifti(filepath, subvolumes[echo_time])

            if echo_time == base_echo_time:
                base_subvolume_filepath = filepath
            else:
                transformation_echo_times.append(echo_time)

            file_data[echo_time] = filepath

        interregistered_dirpath = os.path.join(self.intermediate_save_dir, 'interregistered')

        # Register base image to the target image
        reg = Registration()
        reg.inputs.fixed_image = target
        reg.inputs.moving_image = base_subvolume_filepath
        reg.inputs.output_path = os.path.join(interregistered_dirpath,
                                              '%03d' % base_echo_time)
        reg.inputs.parameters = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]

        if mask is not None:
            reg.inputs.moving_mask = sni.gaussian_filter(np.asarray(mask, dtype=np.float32))

        transformation = reg.run()

        # Load the transformation file. Apply same transform to the remaining images
        for echo_time in transformation_echo_times:
            filename = file_data[echo_time]
            reg = ApplyWarp()
            reg.inputs.moving_image = filename
            reg.inputs.transform_file = transformation[0]
            reg.output_path = io_utils.check_dir(os.path.join(interregistered_dirpath,
                                                              '%03d' % echo_time))
            reg.run()

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def generate_t2_star_map(self):
        svs = []
        spin_lock_times = []
        original_shape = None
        for spin_lock_time in self.subvolumes.keys():
            spin_lock_times.append(spin_lock_time)
            sv = self.subvolumes[spin_lock_time]

            if original_shape is None:
                original_shape = sv.shape
            else:
                assert(sv.shape == original_shape)

            svs.append(sv.reshape((1, -1)))

        vals = np.zeros(svs[0].shape)
        svs = np.concatenate(svs)

        for i in range(len(vals.shape[1])):
            vals[i] = qv.fit_mono_exp(spin_lock_times, svs[..., i], p0=(1.0, 30.0))

        return vals.reshape(original_shape)

    def load_data(self, load_dirpath):
        pass

    def save_data(self, save_dirpath):
        pass


if __name__ == '__main__':
    cq = Cones('../dicoms/healthy07/009', 'dcm')