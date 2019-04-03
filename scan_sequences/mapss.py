import os

from natsort import natsorted
from nipype.interfaces.elastix import Registration

import file_constants as fc
from data_io import format_io_utils as fio_utils
from data_io.format_io import ImageDataFormat
from data_io.nifti_io import NiftiReader, NiftiWriter
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from scan_sequences.scans import NonTargetSequence
from utils import io_utils
from utils import quant_vals as qv
from utils.fits import MonoExponentialFit
from tissues.tissue import Tissue


__EXPECTED_NUM_ECHO_TIMES__ = 4
__R_SQUARED_THRESHOLD__ = 0.9

__INITIAL_T1_RHO_VAL__ = 70.0
__T1_RHO_LOWER_BOUND__ = 0
__T1_RHO_UPPER_BOUND__ = 500

__INITIAL_T2_VAL__ = 30.0
__T2_LOWER_BOUND__ = 0
__T2_UPPER_BOUND__ = 100

__DECIMAL_PRECISION__ = 3


class MAPSS(NonTargetSequence):
    NAME = 'mapss'

    def __init__(self, dicom_path=None, load_path=None):
        self.echo_times = None
        super().__init__(dicom_path=dicom_path, load_path=load_path)

    def __load_dicom__(self):
        super().__load_dicom__()
        self.echo_times = [float(x.headers[0].EchoTime) for x in self.volumes]

    def interregister(self, target_path, mask_path=None,
                      parameter_files=[fc.MAPSS_ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.MAPSS_ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE]):
        num_volumes = len(self.volumes)

        # write all files in nifti format to temp folder for registration
        temp_orig_volume_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'original_volume'))
        nifti_writer = NiftiWriter()
        echo_filepaths = []
        for i in range(num_volumes):
            e_filepath = os.path.join(temp_orig_volume_dirpath, '%03d.nii.gz' % i)
            echo_filepaths.append(e_filepath)
            nifti_writer.save(self.volumes[i], e_filepath)

        # use first echo for registration
        base_index = 0
        base_image = echo_filepaths[base_index]
        other_image_files = []
        for i in range(num_volumes):
            if i == base_index:
                continue
            other_image_files.append((echo_filepaths[i], i))

        temp_interregistered_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'interregistered'))

        print('')
        print('==' * 40)
        print('Interregistering...')
        print('Target: %s' % target_path)
        if mask_path is not None:
            print('Mask: %s' % mask_path)
        print('==' * 40)

        warped_file, transformation_files = self.__interregister_base_file__((base_image, base_index),
                                                                             target_path,
                                                                             temp_interregistered_dirpath,
                                                                             mask_path=mask_path,
                                                                             parameter_files=parameter_files)
        warped_files = dict()
        warped_files[base_index] = warped_file

        nifti_reader = NiftiReader()

        # Load the transformation file. Apply same transform to the remaining images
        for filename, echo_index in other_image_files:
            warped_file = self.__apply_transform__((filename, echo_index),
                                                   transformation_files,
                                                   temp_interregistered_dirpath)
            # append the last warped file - this has all the transforms applied
            warped_files[echo_index] = warped_file

        # copy each of the interregistered warped files to their own output
        volumes = []
        for echo_ind in range(num_volumes):
            volumes.append(nifti_reader.load(warped_files[echo_ind]))

        self.volumes = volumes

    def generate_t1_rho_map(self, tissue: Tissue=None):
        """Generate 3D T1-rho map and r2 fit map using monoexponential fit across subvolumes acquired at different
                echo times
        :param tissue: A Tissue instance
        :return: a T1Rho instance
        """
        echo_inds = range(4)
        bounds = (__T1_RHO_LOWER_BOUND__, __T1_RHO_UPPER_BOUND__),
        tc0 = __INITIAL_T1_RHO_VAL__,
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(echo_inds, tissue, bounds, tc0, decimal_precision)

        return qv_map

    def generate_t2_map(self, tissue: Tissue=None):
        """ Generate 3D T2 map
        :param tissue: a Tissue instance
        :return a T2 instance
        """
        echo_inds = [0, 4, 5, 6]
        bounds = (__T2_LOWER_BOUND__, __T2_UPPER_BOUND__),
        tc0 = __INITIAL_T2_VAL__,
        decimal_precision = __DECIMAL_PRECISION__

        qv_map = self.__fitting_helper(echo_inds, tissue, bounds, tc0, decimal_precision)

        return qv_map

    def __fitting_helper(self, echo_inds, tissue, bounds, tc0, decimal_precision):
        echo_info = [(self.echo_times[i], self.volumes[i]) for i in echo_inds]

        # sort by echo time
        echo_info = sorted(echo_info, key=lambda x: x[0])

        xs = [et for et, _ in echo_info]
        ys = [vol for _, vol in echo_info]

        # only calculate for focused region if a mask is available, this speeds up computation
        mask = tissue.get_mask()
        mef = MonoExponentialFit(xs, ys,
                                 mask=mask,
                                 bounds=bounds,
                                 tc0=tc0,
                                 decimal_precision=decimal_precision)
        qv_map, r2 = mef.fit()

        quant_val_map = qv.T1Rho(qv_map)
        quant_val_map.add_additional_volume('r2', r2)

        tissue.add_quantitative_value(quant_val_map)

        return quant_val_map

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        super().save_data(base_save_dirpath, data_format=data_format)
        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # Save interregistered files
        interregistered_dirpath = os.path.join(base_save_dirpath, 'interregistered')

        num_volumes = len(self.volumes)
        for volume_ind in range(num_volumes):
            nii_filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % volume_ind)
            filepath = fio_utils.convert_format_filename(nii_filepath, data_format)

            self.volumes[volume_ind].save_volume(filepath)

    def load_data(self, base_load_dirpath: str):
        super().load_data(base_load_dirpath)
        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(base_load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['echo_times'])
        return var_names


if __name__ == '__main__':
    scan = MAPSS(dicom_path='../dicoms/mapss_eg')
