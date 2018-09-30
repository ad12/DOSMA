import os

from scan_sequences.scans import NonTargetSequence
from utils import dicom_utils, io_utils
from utils import quant_vals as qv
from nipype.interfaces.elastix import Registration, ApplyWarp
import numpy as np
import file_constants as fc
import scipy.ndimage as sni
from natsort import natsorted
import re

__EXPECTED_NUM_ECHO_TIMES__ = 4

__R_SQUARED_THRESHOLD__ = 0.9
__INITIAL_P0_VALS__ = (1.0, 1/30.0)

__T2_STAR_LOWER_BOUND__ = 0
__T2_STAR_UPPER_BOUND__ = np.inf
__T2_STAR_DECIMAL_PRECISION__ = 3


class Cones(NonTargetSequence):
    NAME = 'cones'

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        super().__init__(dicom_path, dicom_ext)

        self.t2star_map = None
        self.subvolumes = None
        self.focused_mask_filepath = None
        self.echo_times = []

        if load_path:
            self.load_data(load_path)

        if dicom_path is not None:
            self.subvolumes, self.echo_times = self.__split_volumes__(__EXPECTED_NUM_ECHO_TIMES__)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or load_path must be specified')

    def __load_dicom__(self):
        super().__load_dicom__()

        echo_times = []
        for ref in self.refs_dicom:
            echo_times.append(float(ref.EchoTime))

        self.echo_times = natsorted(list(set(echo_times)))

    def interregister(self, target_path, mask_path=None):
        temp_raw_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'raw'))
        subvolumes = self.subvolumes

        raw_filepaths = dict()

        echo_time_inds = natsorted(list(subvolumes.keys()))
        print(echo_time_inds)

        for i in range(len(echo_time_inds)):
            raw_filepath = os.path.join(temp_raw_dirpath, '%03d.nii.gz' % i)
            io_utils.save_nifti(raw_filepath,
                                subvolumes[i],
                                self.pixel_spacing)
            raw_filepaths[i] = raw_filepath

        # last echo should be base
        base_echo_time, base_image = len(echo_time_inds) - 1, raw_filepaths[len(echo_time_inds) - 1]

        temp_interregistered_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'interregistered'))

        print('')
        print('==' * 40)
        print('Interregistering...')
        print('Target: %s' % target_path)
        if mask_path is not None:
            print('Mask: %s' % mask_path)
        print('==' * 40)

        # get number of slices
        volume_shape = self.subvolumes[base_echo_time].shape

        # Register base image to the target image
        print('Registering %s (base image)' % base_image)
        reg = Registration()
        reg.inputs.fixed_image = target_path
        reg.inputs.moving_image = base_image
        reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_interregistered_dirpath,
                                                                 '%03d' % base_echo_time))
        reg.inputs.parameters = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]

        if mask_path is not None:
            #raise ValueError('Moving mask not supported')

            mask, mask_spacing = io_utils.load_nifti(mask_path)

            fixed_mask = np.asarray(sni.gaussian_filter(np.asarray(mask, dtype=np.float32), sigma=3.0) > 0.2,
                                    dtype=np.int8)
            fixed_mask_filepath = os.path.join(temp_interregistered_dirpath, 'dilated-mask.nii.gz')
            io_utils.save_nifti(fixed_mask_filepath, fixed_mask, mask_spacing)
            reg.inputs.fixed_mask = fixed_mask_filepath

        reg.terminal_output = fc.NIPYPE_LOGGING
        reg_output = reg.run()
        reg_output = reg_output.outputs
        transformation_files = reg_output.transform
        warped_files = [(base_echo_time, reg_output.warped_file)]

        print(raw_filepaths)
        files = []
        for echo_time_ind in raw_filepaths.keys():
            filepath = raw_filepaths[echo_time_ind]
            files.append((echo_time_ind, filepath))

        # Load the transformation file. Apply same transform to the remaining images
        for echo_time_ind, filename in files:
            print('Applying transform %s' % filename)
            warped_file = ''
            for f in transformation_files:
                reg = ApplyWarp()
                reg.inputs.moving_image = filename

                reg.inputs.transform_file = f
                reg.inputs.output_path = io_utils.check_dir(os.path.join(temp_interregistered_dirpath,
                                                                  '%03d' % echo_time_ind))
                reg.terminal_output = fc.NIPYPE_LOGGING
                reg_output = reg.run()
                warped_file = reg_output.outputs.warped_file

            assert warped_file != ''

            # append the last warped file - this has all the transforms applied
            warped_files.append((echo_time_ind, warped_file))

        # copy each of the interregistered warped files to their own output
        subvolumes = dict()
        for echo_time_ind, warped_file in warped_files:
            subvolumes[echo_time_ind], _ = io_utils.load_nifti(warped_file)

        self.subvolumes = subvolumes

    def save_data(self, save_dirpath):
        super().save_data(save_dirpath)
        save_dirpath = self.__save_dir__(save_dirpath)

        if self.t2star_map is not None:
            data = {'data': self.t2star_map}
            io_utils.save_h5(os.path.join(save_dirpath, '%s.h5' % qv.QuantitativeValue.T2_STAR.name.lower()), data)
            io_utils.save_nifti(os.path.join(save_dirpath, '%s.nii.gz' % qv.QuantitativeValue.T2_STAR.name.lower()),
                                self.t2star_map, self.pixel_spacing)

        # Save interregistered files
        interregistered_dirpath = os.path.join(save_dirpath, 'interregistered')

        for spin_lock_time in self.subvolumes.keys():
            filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % spin_lock_time)
            io_utils.save_nifti(filepath, self.subvolumes[spin_lock_time], self.pixel_spacing)

    def generate_t2_star_map(self):
        svs = []
        spin_lock_times = []
        original_shape = None

        if self.focused_mask_filepath:
            print('Using focused mask: %s' % self.focused_mask_filepath)
            msk, _ = io_utils.load_nifti(self.focused_mask_filepath)
            msk = msk.reshape(1, -1)

        for spin_lock_time in self.subvolumes.keys():
            spin_lock_times.append(spin_lock_time)
            sv = self.subvolumes[spin_lock_time]

            if original_shape is None:
                original_shape = sv.shape
            else:
                assert (sv.shape == original_shape)

            svr = sv.reshape((1, -1))
            if self.focused_mask_filepath:
                svr = svr * msk

            svs.append(svr)

        svs = np.concatenate(svs)
        vals, r_squared = qv.fit_monoexp_tc(spin_lock_times, svs, __INITIAL_P0_VALS__)

        map_unfiltered = vals.reshape(original_shape)
        r_squared = r_squared.reshape(original_shape)

        t2star_map = map_unfiltered * (r_squared > __R_SQUARED_THRESHOLD__)

        # Filter calculated T1-rho values that are below 0ms and over 100ms
        t2star_map[t2star_map <= __T2_STAR_LOWER_BOUND__] = np.nan
        t2star_map = np.nan_to_num(t2star_map)
        t2star_map[t2star_map > __T2_STAR_UPPER_BOUND__] = np.nan
        t2star_map = np.nan_to_num(t2star_map)

        t2star_map = np.around(t2star_map, __T2_STAR_DECIMAL_PRECISION__)

        self.t2star_map = t2star_map

        return t2star_map

    def load_data(self, load_dirpath):
        super().load_data(load_dirpath)
        load_dirpath = self.__save_dir__(load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __load_interregistered_files__(self, interregistered_dirpath):
        if 'interregistered' not in interregistered_dirpath:
            raise ValueError('Invalid path for loading %s interregistered files' % self.NAME)

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError('No interregistered files found')

        spin_lock_times = []
        subvolumes = []
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError('%s is not an interregisterd \'.gz.nii\' file.' % subfile)

            subfile_num = float(subfile_nums[0])
            spin_lock_times.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume_arr, self.pixel_spacing = io_utils.load_nifti(filepath)
            subvolumes.append(subvolume_arr)

        assert len(spin_lock_times) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError('No interregistered files found')

        subvolumes_dict = dict()
        for i in range(len(spin_lock_times)):
            subvolumes_dict[spin_lock_times[i]] = subvolumes[i]

        return subvolumes_dict

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['echo_times'])

        return var_names


if __name__ == '__main__':
    cq = Cones('../dicoms/healthy07/009', 'dcm', './')