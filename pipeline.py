"""
Main file for scan pipeline - handle argparse
"""
import argparse
import os

from models.get_model import SUPPORTED_MODELS
from scan_sequences.dess import Dess
from models.get_model import get_model
from tissues.femoral_cartilage import FemoralCartilage


DICOM_KEY = 'dicom'
MASK_KEY = 'mask'
SAVE_KEY = 'save'
EXT_KEY = 'ext'
GPU_KEY = 'gpu'

SCAN_KEY = 'scan'
DESS_SCAN_KEY = 'dess'
CUBEQUANT_SCAN_KEYS = ['cubequant', 'cq']
CONES_KEY = 'cones'

T2_KEY = 't2'
T1_RHO_Key = 't1rho'
T2_STAR_KEY = 't2star'

ACTION_KEY = 'action'

SEGMENTATION_MODEL_KEY = 'model'
SEGMENTATION_WEIGHTS_DIR_KEY = 'weights-dir'
SEGMENTATION_BATCH_SIZE_KEY = 'batch-size'

TISSUES_KEY = 'tissues'

def add_segmentation_subparser(parser):
    parser_segment = parser.add_parser('segment')
    parser_segment.add_argument('--%s' % SEGMENTATION_MODEL_KEY, choices=SUPPORTED_MODELS, nargs=1)
    parser_segment.add_argument('--%s' % SEGMENTATION_WEIGHTS_DIR_KEY, type=str, nargs=1,
                                     help='path to directory with weights')
    parser_segment.add_argument('--%s' % SEGMENTATION_BATCH_SIZE_KEY, metavar='B', type=int, default=32, nargs='?',
                                help='batch size for inference. Default: 32')


def handle_segmentation(vargin, scan):
    tissues = vargin['tissues']

    for tissue in tissues:
        tissue.find_weights(vargin[SEGMENTATION_WEIGHTS_DIR_KEY])
        # Load model
        dims = scan.get_dimensions()
        input_shape = (dims[0], dims[1], 1)
        model = get_model(vargin[SEGMENTATION_MODEL_KEY],
                          input_shape=input_shape,
                          weights_path=tissue.weights_filepath)
        model.batch_size = vargin[SEGMENTATION_BATCH_SIZE_KEY]
        scan.segment(model, tissue)


def handle_t2_analysis(scan):
    scan.generate_t2_map()


def handle_dess(vargin):
    scan = Dess(dicom_path=vargin[DICOM_KEY], dicom_ext=vargin[EXT_KEY])
    if (vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'segment'):
        handle_segmentation(vargin, scan)

    if vargin[T2_KEY]:
        handle_t2_analysis(scan)

    return scan


def handle_cubequant(vargin):
    pass


def handle_cones(vargin):
    pass


def parse_args():
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='PROG',
                                     description='Pipeline for segmenting MRI knee volumes',
                                     epilog='NOTE: by default all tissues will be segmented unless specific flags are provided')

    # Dicom and results paths
    parser.add_argument('-d', '--%s' % DICOM_KEY, metavar='D', type=str, nargs=1,
                        help='path to directory storing dicom files')
    parser.add_argument('-m', '--%s' % MASK_KEY, metavar='M', type=str, default='', nargs='?',
                        help='path to segmented mask')
    parser.add_argument('-s', '--%s' % SAVE_KEY, metavar='S', type=str, default='', nargs='?',
                        help='path to directory to save mask. Default: D')
    # If user wants to filter by extension, allow them to specify extension
    parser.add_argument('-e', '--%s' % EXT_KEY, metavar='E', type=str, default=None, nargs='?',
                        help='extension of dicom files')
    parser.add_argument('--%s' % GPU_KEY, metavar='G', type=str, default=None, nargs='?', help='gpu id to use')

    subparsers = parser.add_subparsers(help='sub-command help', dest=SCAN_KEY)

    # DESS parser
    parser_DESS = subparsers.add_parser(DESS_SCAN_KEY, help='analyze DESS sequence')
    parser_DESS.add_argument('-%s' % T2_KEY, action='store_const', default=False, const=True, help='do t2 analysis')
    subparsers_DESS = parser_DESS.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    add_segmentation_subparser(subparsers_DESS)
    parser_DESS.set_defaults(func=handle_dess)

    # Cubequant parser
    parser_cubequant = subparsers.add_parser(CUBEQUANT_SCAN_KEYS[0],
                                             help='analyze cubequant sequence',
                                             aliases=CUBEQUANT_SCAN_KEYS[1:])
    parser_cubequant.add_argument('-%s' % T1_RHO_Key,
                                  action='store_const',
                                  default=False,
                                  const=True,
                                  help='do t1-rho analysis')

    # Cones parser
    parser_cubequant = subparsers.add_parser(CONES_KEY, help='analyze cones sequence')
    parser_cubequant.add_argument('-%s' % T2_STAR_KEY, action='store_const', default=False, const=True,
                                  help='do t2* analysis')

    args = parser.parse_args()
    vargin = vars(args)

    gpu = vargin[GPU_KEY]

    print(vargin)
    # Only supporting femoral cartilage for now

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    try:
        dicom_path = vargin[DICOM_KEY][0]
    except Exception:
        raise ValueError("No path to dicom provided")

    save_path = vargin[SAVE_KEY]
    if save_path == '':
        save_path = dicom_path
        vargin[SAVE_KEY] = save_path

    try:
        if not os.path.isdir(dicom_path):
            raise ValueError
    except ValueError:
        raise NotADirectoryError("Directory \'%s\' does not exist" % dicom_path)


    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    vargin['tissues'] = [FemoralCartilage()]
    # Call func for specific scan (dess, cubequant, cones, etc)
    args.func(vargin)


if __name__ == '__main__':
    parse_args()
