"""
Main file for scan pipeline - handle argparse
"""
import argparse
import os, time

from models.get_model import SUPPORTED_MODELS
from scan_sequences.dess import Dess
from scan_sequences.cube_quant import CubeQuant
from models.get_model import get_model
from tissues.femoral_cartilage import FemoralCartilage
from tissues.patellar_cartilage import PatellarCartilage

from utils.quant_vals import QuantitativeValue as QV, get_qv

from utils import io_utils

from file_constants import DEBUG

import defaults

SUPPORTED_TISSUES = [FemoralCartilage()]
SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]


DICOM_KEY = 'dicom'
MASK_KEY = 'mask'
SAVE_KEY = 'save'
LOAD_KEY = 'load'
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
SEGMENTATION_WEIGHTS_DIR_KEY = 'weights_dir'
SEGMENTATION_BATCH_SIZE_KEY = 'batch_size'

TARGET_SCAN_KEY = 'ts'
TARGET_MASK_KEY = 'tm'
INTERREGISTERED_FILES_DIR_KEY = 'd'

TISSUES_KEY = 'tissues'


def parse_tissues(vargin):
    tissues = []
    for tissue in SUPPORTED_TISSUES:
        if tissue.STR_ID in vargin.keys() and vargin[tissue.STR_ID] and tissue not in tissues:
            load_path = vargin[LOAD_KEY]
            if load_path:
                tissue.load_data(load_path)

            tissues.append(tissue)

    return tissues


def add_segmentation_subparser(parser):
    parser_segment = parser.add_parser('segment')
    parser_segment.add_argument('--%s' % SEGMENTATION_MODEL_KEY, choices=SUPPORTED_MODELS, nargs='?', default='unet2d')
    parser_segment.add_argument('--%s' % SEGMENTATION_WEIGHTS_DIR_KEY, type=str, nargs=1,
                                     help='path to directory with weights')
    parser_segment.add_argument('--%s' % SEGMENTATION_BATCH_SIZE_KEY, metavar='B', type=int,
                                default=defaults.DEFAULT_BATCH_SIZE, nargs='?',
                                help='batch size for inference. Default: 32')
    for tissue in SUPPORTED_TISSUES:
        parser_segment.add_argument('-%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                                   help='handle %s' % tissue.FULL_NAME)


def add_interregister_subparser(parser):
    parser_interregister = parser.add_parser('interregister')
    parser_interregister.add_argument('-%s' % TARGET_SCAN_KEY,
                                      type=str,
                                      nargs=1,
                                      help='path to target image (nifti)')
    parser_interregister.add_argument('-%s' % TARGET_MASK_KEY,
                                      type=str,
                                      nargs='?',
                                      default=None,
                                      help='path to target mask (nifti)')


def handle_tissues(vargin):
    tissues = vargin['tissues']

    load_path = vargin[LOAD_KEY]
    if not load_path:
        ValueError('load path must be specified for handling tissues')

    # Get all supported quantitative values
    qvs = []
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        if vargin[qv.name.lower()]:
            qvs.append((qv, vargin[qv.name.lower()]))

    for tissue in tissues:
        tissue.load_data(load_path)

        for qv, filepath in qvs:
            # load file
            tmp = io_utils.load_h5(filepath)
            qv_map = tmp['data']

            tissue.calc_quant_vals(qv_map, qv)


def handle_segmentation(vargin, scan):
    tissues = vargin['tissues']

    print('')

    if len(tissues) == 0:
        raise ValueError('No tissues specified for segmentation')

    for tissue in tissues:
        segment_weights_path = vargin[SEGMENTATION_WEIGHTS_DIR_KEY]
        tissue.find_weights(segment_weights_path)
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


def handle_t1_rho_analysis(scan, load_dir):
    if not load_dir:
        raise ValueError('Must provide %s for directory to masks' % LOAD_KEY)

    print('\nCalculating T1_rho')

    scan.generate_t1_rho_map()


def handle_dess(vargin):
    scan = Dess(dicom_path=vargin[DICOM_KEY], dicom_ext=vargin[EXT_KEY], load_path=vargin[LOAD_KEY])
    if vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'segment':
        handle_segmentation(vargin, scan)

    if vargin[T2_KEY]:
        handle_t2_analysis(scan)

    return scan


def handle_cubequant(vargin):
    scan = CubeQuant(dicom_path=vargin[DICOM_KEY],
                     dicom_ext=vargin[EXT_KEY],
                     load_path=vargin[LOAD_KEY])

    scan.tissues = vargin['tissues']

    if vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'interregister':
        target_scan = vargin[TARGET_SCAN_KEY]
        scan.interregister(target_scan[0], vargin[TARGET_MASK_KEY])

    if vargin[T1_RHO_Key]:
        handle_t1_rho_analysis(scan, vargin[LOAD_KEY])

    return scan


def handle_cones(vargin):
    pass


def save_info(dirpath, scan):
    scan.save_data(dirpath)
    for tissue in scan.tissues:
        tissue.save_data(dirpath)


def parse_args():
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='PROG',
                                     description='Pipeline for segmenting MRI knee volumes')

    # Dicom and results paths
    parser.add_argument('-d', '--%s' % DICOM_KEY, metavar='D', type=str, default=None, nargs='?',
                        help='path to directory storing dicom files')
    parser.add_argument('-s', '--%s' % SAVE_KEY, metavar='S', type=str, default=None, nargs='?',
                        help='path to directory to save mask. Default: D')
    parser.add_argument('-l', '--%s' % LOAD_KEY, metavar='L', type=str, default=None, nargs='?',
                        help='path to data directory')

    # If user wants to filter by extension, allow them to specify extension
    parser.add_argument('-e', '--%s' % EXT_KEY, metavar='E', type=str, default='dcm', nargs='?',
                        help='extension of dicom files')
    parser.add_argument('--%s' % GPU_KEY, metavar='G', type=str, default=None, nargs='?', help='gpu id to use')

    subparsers = parser.add_subparsers(help='sub-command help', dest=SCAN_KEY)

    # DESS parser
    parser_dess = subparsers.add_parser(DESS_SCAN_KEY, help='analyze DESS sequence')
    parser_dess.add_argument('-%s' % T2_KEY, action='store_const', default=False, const=True, help='compute T2 map')
    subparsers_dess = parser_dess.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    add_segmentation_subparser(subparsers_dess)
    parser_dess.set_defaults(func=handle_dess)

    # Cubequant parser
    parser_cubequant = subparsers.add_parser(CUBEQUANT_SCAN_KEYS[0],
                                             help='analyze cubequant sequence',
                                             aliases=CUBEQUANT_SCAN_KEYS[1:])
    parser_cubequant.add_argument('-%s' % T1_RHO_Key,
                                  action='store_const',
                                  default=False,
                                  const=True,
                                  help='do t1-rho analysis')

    subparsers_cubequant = parser_cubequant.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    add_interregister_subparser(subparsers_cubequant)
    parser_cubequant.set_defaults(func=handle_cubequant)

    # Tissue parser
    parser_tissue = subparsers.add_parser(TISSUES_KEY, help='analyze tissues')
    for tissue in SUPPORTED_TISSUES:
        parser_tissue.add_argument('-%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                                   help='handle %s' % tissue.FULL_NAME)
    qvs_dict = dict()
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        qv_name = qv.name.lower()
        qvs_dict[qv_name] = qv
        parser_tissue.add_argument('-%s' % qv_name, nargs='?', default=None,
                                   help='calculate %s' % qv_name)

        parser_tissue.set_defaults(func=handle_tissues)


    start_time = time.time()
    args = parser.parse_args()
    vargin = vars(args)

    gpu = vargin[GPU_KEY]

    if DEBUG:
        print(vargin)
    # Only supporting femoral cartilage for now

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    dicom_path = vargin[DICOM_KEY]
    load_path = vargin[LOAD_KEY]

    if not dicom_path and not load_path:
        raise ValueError('Must provide path to dicoms or path to load data from')

    save_path = vargin[SAVE_KEY]
    if not save_path:
        save_path = dicom_path if dicom_path else load_path
        vargin[SAVE_KEY] = save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tissues = parse_tissues(vargin)
    vargin['tissues'] = tissues

    # Call func for specific scan (dess, cubequant, cones, etc)
    scan = args.func(vargin)
    save_info(vargin[SAVE_KEY], scan)

    print('Time Elapsed: %0.2f seconds' % (time.time() - start_time))

    #
    # # Cones parser
    # parser_cubequant = subparsers.add_parser(CONES_KEY, help='analyze cones sequence')
    # parser_cubequant.add_argument('-%s' % T2_STAR_KEY, action='store_const', default=False, const=True,
    #                               help='do t2* analysis')


if __name__ == '__main__':
    parse_args()
