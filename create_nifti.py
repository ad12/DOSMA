import argparse
import os
import scipy.io as sio
from utils import dicom_utils
from med_objects.med_volume import MedicalVolume


INPUT_KEY = 'input'
VARIABLE_KEY = 'var'
FILETYPE_KEY = 'filetype'
FILE_KEY = 'file'
MAT_FILE = 'mat'
SAMPLE_DICOM_KEY = 'sample_dicom'


def handle_mat(vargin):
    mat_filepath = vargin[FILE_KEY][0]
    var_name = vargin[VARIABLE_KEY][0]
    sample_dicom_path = vargin[SAMPLE_DICOM_KEY][0]

    # extract array from mat file
    mat_contents = sio.loadmat(mat_filepath)
    arr = mat_contents[var_name]

    # extract pixel spacing from dicom
    pixel_spacing = dicom_utils.get_pixel_spacing(sample_dicom_path)

    v = MedicalVolume(arr, pixel_spacing)

    fs = os.path.splitext(mat_filepath)
    save_filepath = '%s.nii.gz' % fs[0]
    v.save_volume(save_filepath)


def parse_args():
    """Parse arguments given through command line (argv)

        :raises ValueError if dicom path is not provided
        :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='pipeline',
                                     description='Pipeline for segmenting MRI knee volumes')

    subparsers = parser.add_subparsers(help='sub-command help', dest=FILETYPE_KEY)
    parser_mat = subparsers.add_parser(MAT_FILE, help='convert .mat to .nii.gz format')

    # Dicom and results paths
    parser_mat.add_argument('-f', '--%s' % FILE_KEY, metavar='F', type=str, default=None, nargs=1,
                        help='path to .mat file')
    parser_mat.add_argument('-v', '--%s' % VARIABLE_KEY, metavar='V', type=str, default=None, nargs=1,
                            help='if using .mat format, what variable name to analyze')
    parser_mat.add_argument('-sd', '--%s' % SAMPLE_DICOM_KEY, metavar='SD', type=str, default=None, nargs=1,
                            help='sample dicom file to extract spacing from')

    parser_mat.set_defaults(func=handle_mat)

    args = parser.parse_args()
    vargin = vars(args)

    args.func(vargin)


if __name__ == '__main__':
    parse_args()
