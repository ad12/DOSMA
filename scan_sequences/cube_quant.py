import os

from scan_sequences.scans import NonTargetSequence
from natsort import natsorted
from nipype.interfaces.elastix import Registration, ApplyWarp
from utils import io_utils
from time import gmtime, strftime
import scipy.ndimage as sni
import numpy as np
import file_constants as fc
from utils import quant_vals as qv

__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4


class CubeQuant(NonTargetSequence):
    NAME = 'cube_quant'
    TEMP_PATH = os.path.join(fc.TEMP_FOLDER_PATH, '%s', '%s') % (NAME, strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    def __init__(self, dicom_path=None, dicom_ext=None, save_dir=None, target_dict=None, interregistered_volumes_path=''):
        super().__init__(dicom_path, dicom_ext)
        self.subvolumes = None

        if dicom_path is not None:
            self.save_dir = save_dir

            self.subvolumes = self.__split_volumes__()
            self.intermediate_save_dir = os.path.join(self.save_dir, self.NAME)
            self.intraregistered_data = self.__intraregister__(self.subvolumes)

            self.interregister(target_dict['target_filepath'], target_dict['mask_filepath'])

        elif interregistered_volumes_path:
            self.subvolumes = self.__load_interregistered_files__(interregistered_volumes_path)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or interregistered_volumes_path must be specified')

    def interregister(self, target, mask=None):
        base_spin_lock_time, base_image = self.intraregistered_data['BASE']
        files = self.intraregistered_data['FILES']

        # Register base image to the target image
        reg = Registration()
        reg.inputs.fixed_image = target
        reg.inputs.moving_image = base_image
        reg.inputs.output_path = os.path.join(self.intermediate_save_dir,
                                              'interregistered',
                                              '%03d' % base_spin_lock_time)
        reg.inputs.parameters = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]

        if mask is not None:
            reg.inputs.moving_mask = sni.gaussian_filter(np.asarray(mask, dtype=np.float32))

        transformation = reg.run()

        interregistered_files = []
        interregistered_files.append((base_spin_lock_time, base_image))
        interregistered_dirpath = os.path.join(self.save_dir, 'interregistered')

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
        svs = np.concatenate(svs)

        for i in range(len(vals.shape[1])):
            vals[i] = qv.fit_mono_exp(spin_lock_times, svs[..., i], p0=(1.0, 30.0))

        return vals.reshape(original_shape)

    def __split_volumes__(self):
        refs_dicom = self.refs_dicom
        volume = self.volume
        subvolumes_dict = dict()

        # get spin_lock_time time for each refs_dicom
        for i in range(len(refs_dicom)):
            echo_time = refs_dicom[i].EchoTime
            if echo_time in subvolumes_dict.keys():
                subvolumes_dict[echo_time].append(i)
            else:
                subvolumes_dict[echo_time] = [i]

        # assert expected number of echos
        num_spin_lock_times = len(list(subvolumes_dict.keys()))
        assert num_spin_lock_times == __EXPECTED_NUM_SPIN_LOCK_TIMES__

        # number of spin lock times should go evenly into subvolume
        if (len(refs_dicom) % num_spin_lock_times) != 0:
            raise ValueError('Uneven number of dicom files - %d dicoms, %d spin lock times' % (len(refs_dicom), num_spin_lock_times))

        num_slices_per_subvolume = len(refs_dicom) / num_spin_lock_times

        for key in subvolumes_dict.keys():
            if (len(subvolumes_dict[key]) != num_slices_per_subvolume):
                raise ValueError('spin_lock_time \'%s\' has %d slices. Expected %d slices' % (key, len(subvolumes_dict[key]), num_slices_per_subvolume))

        for key in subvolumes_dict.keys():
            sv = subvolumes_dict[key]
            sv = volume[..., sv]
            subvolumes_dict[key] = sv

        return subvolumes_dict

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
        raw_volumes_base_path = os.path.join(self.TEMP_PATH, 'raw')

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
            reg.inputs.output_path = io_utils.check_dir(os.path.join(self.TEMP_PATH,
                                                                     'intraregistered',
                                                                     '%03d' % spin_lock_time))
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            transform, warped_file, _, _ = reg.run()

            intraregistered_files.append((spin_lock_time, warped_file))

        return {'BASE': (ordered_spin_lock_times[0], spin_lock_nii_files[0]),
                'FILES': intraregistered_files}

    def __load_interregistered_files__(self, interregistered_dirpath):
        if 'interregistered' not in interregistered_dirpath:
            raise ValueError('Invalid path for loading cube_quant interregistered files')

        subdirs = io_utils.get_subdirs(interregistered_dirpath)
        subdirs = natsorted(subdirs)

        spin_lock_times = []
        subvolumes = []
        for subdir in subdirs:
            spin_lock_times.append(float(subdir))
            filepath = os.path.join(interregistered_dirpath, subdir, 'output.nii')
            subvolume_arr = io_utils.load_nifti(filepath)
            subvolumes.append(subvolume_arr)

        assert len(spin_lock_times) == len(subvolumes), "Number of subvolumes mismatch"

        subvolumes_dict = dict()
        for i in range(len(spin_lock_times)):
            subvolumes_dict[spin_lock_times[i]] = subvolumes[i]

        return subvolumes_dict

    def load_data(self, load_dirpath):
        pass

    def save_data(self, save_dirpath):
        pass

if __name__ == '__main__':
    cq = CubeQuant('../dicoms/healthy07/008', 'dcm')


