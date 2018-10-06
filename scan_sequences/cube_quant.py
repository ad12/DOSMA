import os

from natsort import natsorted
from nipype.interfaces.elastix import Registration

import file_constants as fc
from scan_sequences.scans import NonTargetSequence
from utils import io_utils
from utils import quant_vals as qv
from utils.fits import MonoExponentialFit

__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4
__R_SQUARED_THRESHOLD__ = 0.9
__INITIAL_T1_RHO_VAL__ = 70.0

__T1_RHO_LOWER_BOUND__ = 0
__T1_RHO_UPPER_BOUND__ = 500
__T1_RHO_DECIMAL_PRECISION__ = 3


class CubeQuant(NonTargetSequence):
    NAME = 'cubequant'

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        super().__init__(dicom_path, dicom_ext, load_path=load_path)

        self.t1rho_map = None
        self.r2 = None
        self.subvolumes = None
        self.focused_mask_filepath = None

        if load_path:
            self.load_data(load_path)

        if dicom_path is not None:
            self.subvolumes, self.spin_lock_times = self.__split_volumes__(__EXPECTED_NUM_SPIN_LOCK_TIMES__)
            self.intraregistered_data = self.__intraregister__(self.subvolumes)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or load_path must be specified')

    def interregister(self, target_path, mask_path=None):
        base_spin_lock_time, base_image = self.intraregistered_data['BASE']
        files = self.intraregistered_data['FILES']

        temp_interregistered_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'interregistered'))

        print('')
        print('==' * 40)
        print('Interregistering...')
        print('Target: %s' % target_path)
        if mask_path is not None:
            print('Mask: %s' % mask_path)
        print('==' * 40)

        if not mask_path:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
        else:
            parameter_files = [fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE]

        warped_file, transformation_files = self.__interregister_base_file__((base_image, base_spin_lock_time),
                                                                             target_path,
                                                                             temp_interregistered_dirpath,
                                                                             mask_path=mask_path,
                                                                             parameter_files=parameter_files)
        warped_files = [(base_spin_lock_time, warped_file)]

        # Load the transformation file. Apply same transform to the remaining images
        for spin_lock_time, filename in files:
            warped_file = self.__apply_transform__((filename, spin_lock_time),
                                                   transformation_files,
                                                   temp_interregistered_dirpath)
            # append the last warped file - this has all the transforms applied
            warped_files.append((spin_lock_time, warped_file))

        # copy each of the interregistered warped files to their own output
        subvolumes = dict()
        for spin_lock_time, warped_file in warped_files:
            subvolumes[spin_lock_time] = io_utils.load_nifti(warped_file)

        self.subvolumes = subvolumes

    def generate_t1_rho_map(self):
        spin_lock_times = []
        subvolumes_list = []
        msk = None
        if self.focused_mask_filepath:
            print('Using focused mask: %s' % self.focused_mask_filepath)
            msk = io_utils.load_nifti(self.focused_mask_filepath)

        sorted_keys = natsorted(list(self.subvolumes.keys()))
        for spin_lock_time_index in sorted_keys:
            subvolumes_list.append(self.subvolumes[spin_lock_time_index])
            spin_lock_times.append(self.spin_lock_times[spin_lock_time_index])

        mef = MonoExponentialFit(spin_lock_times, subvolumes_list,
                                 mask=msk,
                                 bounds=(__T1_RHO_LOWER_BOUND__, __T1_RHO_UPPER_BOUND__),
                                 tc0=__INITIAL_T1_RHO_VAL__,
                                 decimal_precision=__T1_RHO_DECIMAL_PRECISION__)

        self.t1rho_map, self.r2 = mef.fit()

        return self.t1rho_map

    def __intraregister__(self, subvolumes):
        """
        Register subvolumes to each other using affine registration with elastix
        :param subvolumes:
        :return:
        """
        if subvolumes is None:
            raise ValueError('subvolumes must be dict()')

        print('')
        print('==' * 40)
        print('Intraregistering...')
        print('==' * 40)

        # temporarily save subvolumes as nifti file
        ordered_spin_lock_time_indices = natsorted(list(subvolumes.keys()))
        raw_volumes_base_path = io_utils.check_dir(os.path.join(self.temp_path, 'raw'))

        # Use first spin lock time as a basis for registration
        spin_lock_nii_files = []
        for spin_lock_time_index in ordered_spin_lock_time_indices:
            filepath = os.path.join(raw_volumes_base_path, '%03d' % spin_lock_time_index + '.nii.gz')
            spin_lock_nii_files.append(filepath)

            subvolumes[spin_lock_time_index].save_volume(filepath)

        target_filepath = spin_lock_nii_files[0]

        intraregistered_files = []
        for i in range(1, len(spin_lock_nii_files)):
            spin_file = spin_lock_nii_files[i]
            spin_lock_time_index = ordered_spin_lock_time_indices[i]

            reg = Registration()
            reg.inputs.fixed_image = target_filepath
            reg.inputs.moving_image = spin_file
            reg.inputs.output_path = io_utils.check_dir(os.path.join(self.temp_path,
                                                                     'intraregistered',
                                                                     '%03d' % spin_lock_time_index))
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            reg.terminal_output = fc.NIPYPE_LOGGING
            print('Registering %s --> %s' % (str(spin_lock_time_index), str(ordered_spin_lock_time_indices[0])))
            tmp = reg.run()

            warped_file = tmp.outputs.warped_file
            intraregistered_files.append((spin_lock_time_index, warped_file))

        return {'BASE': (ordered_spin_lock_time_indices[0], spin_lock_nii_files[0]),
                'FILES': intraregistered_files}

    def save_data(self, save_dirpath):
        super().save_data(save_dirpath)
        save_dirpath = self.__save_dir__(save_dirpath)

        if self.t1rho_map is not None:
            assert (self.r2 is not None)

            t1rho_map_filepath = os.path.join(save_dirpath, '%s.nii.gz' % qv.QuantitativeValues.T1_RHO.name.lower())
            self.t1rho_map.save_volume(t1rho_map_filepath)

            t1rho_r2_map_filepath = os.path.join(save_dirpath,
                                                 '%s_r2.nii.gz' % qv.QuantitativeValues.T1_RHO.name.lower())
            self.r2.save_volume(t1rho_r2_map_filepath)

        # Save interregistered files
        interregistered_dirpath = os.path.join(save_dirpath, 'interregistered')

        for spin_lock_time_index in self.subvolumes.keys():
            filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % spin_lock_time_index)
            self.subvolumes[spin_lock_time_index].save_volume(filepath)

    def load_data(self, load_dirpath):
        super().load_data(load_dirpath)
        load_dirpath = self.__save_dir__(load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['spin_lock_times'])
        return var_names
