import os

from scan_sequences.scans import NonTargetSequence
from natsort import natsorted
from nipype.interfaces.elastix import Registration, ApplyWarp
from utils import io_utils
import scipy.ndimage as sni
import numpy as np
import file_constants as fc
from utils import quant_vals as qv

__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4
__R_SQUARED_THRESHOLD__ = 0.9
__INITIAL_P0_VALS__ = (1.0, 30.0)

class CubeQuant(NonTargetSequence):
    NAME = 'cube_quant'

    def __init__(self, dicom_path=None, dicom_ext=None, save_dir=None, interregistered_volumes_path=''):
        super().__init__(dicom_path, dicom_ext)
        self.subvolumes = None

        if dicom_path is not None:
            self.save_dir = save_dir

            self.subvolumes = self.__split_volumes__(__EXPECTED_NUM_SPIN_LOCK_TIMES__)
            self.intermediate_save_dir = os.path.join(self.save_dir, self.NAME)
            self.intraregistered_data = self.__intraregister__(self.subvolumes)

        elif interregistered_volumes_path:
            self.subvolumes = self.__load_interregistered_files__(interregistered_volumes_path)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or interregistered_volumes_path must be specified')

    def interregister(self, target, mask=None):
        base_spin_lock_time, base_image = self.intraregistered_data['BASE']
        files = self.intraregistered_data['FILES']

        interregistered_dirpath = os.path.join(self.intermediate_save_dir, 'interregistered')

        # Register base image to the target image
        reg = Registration()
        reg.inputs.fixed_image = target
        reg.inputs.moving_image = base_image
        reg.inputs.output_path = os.path.join(self.intermediate_save_dir,
                                              '%03d' % base_spin_lock_time)
        reg.inputs.parameters = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]

        if mask is not None:
            reg.inputs.moving_mask = sni.gaussian_filter(np.asarray(mask, dtype=np.float32))

        transformation = reg.run()

        # Load the transformation file. Apply same transform to the remaining images
        for spin_lock_time, filename in files:
            reg = ApplyWarp()
            reg.inputs.moving_image = filename
            reg.inputs.transform_file = transformation[0]
            reg.output_path = io_utils.check_dir(os.path.join(interregistered_dirpath,
                                                              '%03d' % spin_lock_time))
            reg.run()

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def generate_t1_rho_map(self):
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
        r_squared = np.zeros(svs[0].shape)
        svs = np.concatenate(svs)

        for i in range(len(vals.shape[1])):
            vals[i], r_squared[i] = qv.fit_mono_exp(spin_lock_times, svs[..., i], p0=__INITIAL_P0_VALS__)

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        return map_unfiltered * (r_squared > __R_SQUARED_THRESHOLD__)

    def __intraregister__(self, subvolumes):
        """
        Register subvolumes to each other using affine registration with elastix
        :param subvolumes:
        :return:
        """
        if subvolumes is None:
            raise ValueError('subvolumes must be dict()')

        # temporarily save subvolumes as nifti file
        ordered_spin_lock_times = natsorted(list(subvolumes.keys()))
        raw_volumes_base_path = os.path.join(self.temp_path, 'raw')

        # Use first spin lock time as a basis for registration
        spin_lock_nii_files = []
        count = 1
        for spin_lock_time in ordered_spin_lock_times:
            filepath = os.path.join(raw_volumes_base_path, '%03d' % count + '.nii')
            spin_lock_nii_files.append(filepath)

            io_utils.save_nifti(filepath, subvolumes[spin_lock_time])

            count += 1

        target_filepath = spin_lock_nii_files[0]

        intraregistered_files = []
        for i in range(1,len(spin_lock_nii_files)):
            spin_file = spin_lock_nii_files[i]
            spin_lock_time = ordered_spin_lock_times[i]

            reg = Registration()
            reg.inputs.fixed_image = target_filepath
            reg.inputs.moving_image = spin_file
            reg.inputs.output_path = io_utils.check_dir(os.path.join(self.temp_path,
                                                                     'intraregistered',
                                                                     '%03d' % spin_lock_time))
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            transform, warped_file, _, _ = reg.run()

            intraregistered_files.append((spin_lock_time, warped_file))

        return {'BASE': (ordered_spin_lock_times[0], spin_lock_nii_files[0]),
                'FILES': intraregistered_files}

    def load_data(self, load_dirpath):
        pass

    def save_data(self, save_dirpath):
        pass

if __name__ == '__main__':
    cq = CubeQuant('../dicoms/healthy07/008', 'dcm')


