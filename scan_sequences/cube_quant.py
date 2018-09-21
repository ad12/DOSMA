import os

from scan_sequences.scans import NonTargetSequence
from utils import dicom_utils
from natsort import natsorted
from nipype.interfaces.elastix import Registration
from utils import io_utils
import shutil
from time import gmtime, strftime
import SimpleITK as sitk


__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4


class CubeQuant(NonTargetSequence):
    NAME = 'cube_quant'

    def __init__(self, dicom_path=None, dicom_ext=None, save_dir=None, intraregistered_volumes_path='', interregistered_volumes_path=''):
        super().__init__(dicom_path, dicom_ext)
        self.spin_lock_times = None
        self.subvolumes = None

        self.split_volumes()
        self.save_dir = save_dir
        self.intraregister(self.subvolumes)

    def split_volumes(self):
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

        self.subvolumes = subvolumes_dict

    def intraregister(self, subvolumes):
        """
        Register subvolumes to each other using affine registration with elastix
        :param subvolumes:
        :return:
        """
        if subvolumes is None:
            raise ValueError('subvolumes must be dict()')
        # temporarily save subvolumes as nifti file
        ordered_spin_lock_times = natsorted(list(subvolumes.keys()))
        temp_base_path = './temp-%s' % (strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

        # Use first spin lock time as a basis for registration
        spin_lock_nii_files = []
        count = 1
        for spin_lock_time in ordered_spin_lock_times:
            filepath = os.path.join(temp_base_path, '%03d' % count + '.nii')
            spin_lock_nii_files.append(filepath)

            io_utils.save_nifti(filepath, subvolumes[spin_lock_time])

            count += 1

        target_filepath = spin_lock_nii_files[0]


        for i in range(1,len(spin_lock_nii_files)):
            spin_file = spin_lock_nii_files[i]

            reg = Registration()
            reg.inputs.fixed_image = target_filepath
            reg.inputs.moving_image = spin_file
            reg.inputs.parameters = ['/Users/arjundesai/Documents/stanford/research/msk_pipeline_raw/elastix_params/parameters-affine.txt']
            reg.run()

            sitk.Simple

    def interregister(self, target):
        pass

    def load_data(self, load_dirpath):
        pass

    def save_data(self, save_dirpath):
        pass

if __name__ == '__main__':
    cq = CubeQuant('../dicoms/healthy07/008', 'dcm')


