import os

import numpy as np
from natsort import natsorted

import file_constants as fc
from data_io.format_io import ImageDataFormat
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from scan_sequences.scans import NonTargetSequence
from utils import io_utils
from utils import quant_vals as qv
from utils.fits import MonoExponentialFit

__EXPECTED_NUM_ECHO_TIMES__ = 4

__INITIAL_T2_STAR_VAL__ = 30.0

__T2_STAR_LOWER_BOUND__ = 0
__T2_STAR_UPPER_BOUND__ = np.inf
__T2_STAR_DECIMAL_PRECISION__ = 3


class Cones(NonTargetSequence):
    """Handles analysis for Cones scan sequence """
    NAME = 'cones'

    def __init__(self, dicom_path=None, load_path=None):
        raise NotImplementedError('%s currently not supported') % self.NAME

        self.subvolumes = None
        self.echo_times = []
        super().__init__(dicom_path=dicom_path, load_path=load_path)

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

        for i in range(len(echo_time_inds)):
            raw_filepath = os.path.join(temp_raw_dirpath, '%03d.nii.gz' % i)
            subvolumes[i].save_volume(raw_filepath)
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

        files_to_warp = []
        for echo_time_ind in raw_filepaths.keys():
            if echo_time_ind == base_echo_time:
                continue
            filepath = raw_filepaths[echo_time_ind]
            files_to_warp.append((echo_time_ind, filepath))

        if not mask_path:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
        else:
            parameter_files = [fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE]

        warped_file, transformation_files = self.__interregister_base_file__((base_image, base_echo_time),
                                                                             target_path,
                                                                             temp_interregistered_dirpath,
                                                                             mask_path=mask_path,
                                                                             parameter_files=parameter_files)
        warped_files = [(base_echo_time, warped_file)]

        # Load the transformation file. Apply same transform to the remaining images
        for echo_time, filename in files_to_warp:
            warped_file = self.__apply_transform__((filename, echo_time),
                                                   transformation_files,
                                                   temp_interregistered_dirpath)
            # append the last warped file - this has all the transforms applied
            warped_files.append((echo_time, warped_file))

        # copy each of the interregistered warped files to their own output
        subvolumes = dict()
        for echo_time, warped_file in warped_files:
            subvolumes[echo_time] = io_utils.load_nifti(warped_file)

        self.subvolumes = subvolumes

    def generate_t2_star_map(self, tissues=None):
        """Generate 3D T2* map and r2 fit map using monoexponential fit across subvolumes acquired at different
                echo times
        :param tissues: A list of Tissue instances specifying which tissue to examine
                        if None, use list of tissues class initialized with
        :return A list of T2Star instances
        """
        if tissues is None:
            tissues = self.tissues

        quant_maps = []

        for tissue in tissues:
            msk = tissue.get_mask()
            spin_lock_times = []
            subvolumes_list = []

            for echo_time in self.subvolumes.keys():
                spin_lock_times.append(echo_time)
                subvolumes_list.append(self.subvolumes[echo_time])

            mef = MonoExponentialFit(spin_lock_times,
                                     subvolumes_list,
                                     mask=msk,
                                     bounds=(__T2_STAR_LOWER_BOUND__, __T2_STAR_UPPER_BOUND__),
                                     tc0=__INITIAL_T2_STAR_VAL__,
                                     decimal_precision=__T2_STAR_DECIMAL_PRECISION__)

            t2star_map, r2 = mef.fit()

            quant_val_map = qv.T2Star(t2star_map)
            quant_val_map.add_additional_volume('r2', r2)

            quant_maps.append(quant_val_map)

            tissue.add_quantitative_value(quant_val_map)

        return quant_maps

    def save_data(self, base_save_dirpath, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        super().save_data(base_save_dirpath, data_format=data_format)
        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # Save interregistered files
        interregistered_dirpath = os.path.join(base_save_dirpath, 'interregistered')

        for spin_lock_time in self.subvolumes.keys():
            filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % spin_lock_time)
            self.subvolumes[spin_lock_time].save_volume(filepath)

    def load_data(self, base_load_dirpath):
        super().load_data(base_load_dirpath)
        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(base_load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['echo_times'])

        return var_names
