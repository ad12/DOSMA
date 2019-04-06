"""
Main file for scan pipeline - handle argparse
"""

import argparse
import os
import time

import defaults
import file_constants as fc
from data_io.format_io import ImageDataFormat, SUPPORTED_FORMATS
from models.get_model import SUPPORTED_MODELS
from models.get_model import get_model
from msk import knee
from scan_sequences.cones import Cones
from scan_sequences.cube_quant import CubeQuant
from scan_sequences.qdess import QDess
from utils.quant_vals import QuantitativeValues as QV

SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]

DEBUG_KEY = 'debug'

DICOM_KEY = 'dicom'
MASK_KEY = 'mask'
SAVE_KEY = 'save'
LOAD_KEY = 'load'
DATA_FORMAT_KEY = 'format'
GPU_KEY = 'gpu'

SCAN_KEY = 'scan'
qDESS_SCAN_KEY = 'qdess'
CUBEQUANT_SCAN_KEYS = ['cubequant', 'cq']
CONES_KEY = 'cones'

T2_KEY = 't2'
T1_RHO_Key = 't1_rho'
T2_STAR_KEY = 't2_star'

ACTION_KEY = 'action'

SEGMENTATION_MODEL_KEY = 'model'
SEGMENTATION_WEIGHTS_DIR_KEY = 'weights_dir'
SEGMENTATION_BATCH_SIZE_KEY = 'batch_size'

TARGET_SCAN_KEY = 'ts'
TARGET_MASK_KEY = 'tm'
INTERREGISTERED_FILES_DIR_KEY = 'd'

ORIENTATION_KEY = 'orientation'

TISSUES_KEY = 'tissues'

USE_RMS_KEY = 'rms'


def parse_tissues(vargin):
    tissues = []
    for tissue in knee.SUPPORTED_TISSUES:
        if tissue.STR_ID in vargin.keys() and vargin[tissue.STR_ID] and tissue not in tissues:
            load_path = vargin[LOAD_KEY]
            if load_path:
                tissue.load_data(load_path)

            tissues.append(tissue)

    # if no tissues are specified, do computation for all supported tissues
    if len(tissues) == 0:
        print('No tissues specified, computing for all supported tissues...')
        tissues = []
        for tissue in knee.SUPPORTED_TISSUES:
            if tissue not in tissues:
                load_path = vargin[LOAD_KEY]
                if load_path:
                    tissue.load_data(load_path)

                tissues.append(tissue)

    analysis_str = 'Tissue(s): '
    for tissue in tissues:
        analysis_str += '%s, ' % tissue.FULL_NAME

    print(analysis_str)

    return tissues


def add_segmentation_subparser(parser):
    parser_segment = parser.add_parser('segment')
    parser_segment.add_argument('--%s' % SEGMENTATION_MODEL_KEY, choices=SUPPORTED_MODELS, nargs='?', default='unet2d',
                                help='Model to use for segmentation. Choices: {%s}' % 'unet2d')
    parser_segment.add_argument('--%s' % SEGMENTATION_WEIGHTS_DIR_KEY, type=str, nargs=1,
                                help='path to directory with weights')
    parser_segment.add_argument('--%s' % SEGMENTATION_BATCH_SIZE_KEY, metavar='B', type=int,
                                default=defaults.DEFAULT_BATCH_SIZE, nargs='?',
                                help='batch size for inference. Default: %d' % defaults.DEFAULT_BATCH_SIZE)

    return parser_segment


def add_interregister_subparser(parser):
    parser_interregister = parser.add_parser('interregister')
    parser_interregister.add_argument('-%s' % TARGET_SCAN_KEY,
                                      type=str,
                                      nargs=1,
                                      help='path to target image. Type: nifti (.nii.gz)')
    parser_interregister.add_argument('-%s' % TARGET_MASK_KEY,
                                      type=str,
                                      nargs='?',
                                      default=None,
                                      help='path to target mask. Type: nifti (.nii.gz)')


def handle_segmentation(vargin, scan):
    tissues = vargin['tissues']

    print('')

    if len(tissues) == 0:
        raise ValueError('No tissues specified for segmentation')

    for tissue in tissues:
        segment_weights_path = vargin[SEGMENTATION_WEIGHTS_DIR_KEY][0]
        tissue.find_weights(segment_weights_path)
        # Load model
        dims = scan.get_dimensions()
        input_shape = (dims[0], dims[1], 1)
        model = get_model(vargin[SEGMENTATION_MODEL_KEY],
                          input_shape=input_shape,
                          weights_path=tissue.weights_filepath)
        model.batch_size = vargin[SEGMENTATION_BATCH_SIZE_KEY]
        scan.segment(model, tissue)


def handle_qdess(vargin):
    print('\nAnalyzing qDESS...')

    tissues = vargin['tissues']

    scan = QDess(dicom_path=vargin[DICOM_KEY], load_path=vargin[LOAD_KEY])

    scan.use_rms = vargin[USE_RMS_KEY] if USE_RMS_KEY in vargin.keys() else False

    if vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'segment':
        handle_segmentation(vargin, scan)

    if vargin[T2_KEY]:
        print('')
        for tissue in tissues:
            print('Calculating T2 - %s' % tissue.FULL_NAME)
            scan.generate_t2_map(tissue)

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    return scan


def handle_cubequant(vargin):
    print('\nAnalyzing cubequant...')
    scan = CubeQuant(dicom_path=vargin[DICOM_KEY],
                     load_path=vargin[LOAD_KEY])

    scan.tissues = vargin['tissues']

    if vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'interregister':
        target_scan = vargin[TARGET_SCAN_KEY][0]
        if not os.path.isfile(target_scan):
            raise FileNotFoundError('%s is not a file' % target_scan)

        scan.interregister(target_scan, vargin[TARGET_MASK_KEY])

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    if vargin[T1_RHO_Key]:
        print('\nCalculating T1_rho')
        scan.generate_t1_rho_map()

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    for tissue in scan.tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    return scan


def handle_cones(vargin):
    print('\nAnalyzing cones...')
    scan = Cones(dicom_path=vargin[DICOM_KEY],
                 load_path=vargin[LOAD_KEY])

    scan.tissues = vargin['tissues']

    if vargin[ACTION_KEY] is not None and vargin[ACTION_KEY] == 'interregister':
        target_scan = vargin[TARGET_SCAN_KEY]
        scan.interregister(target_scan[0], vargin[TARGET_MASK_KEY])

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    load_filepath = vargin[LOAD_KEY] if vargin[LOAD_KEY] else vargin[SAVE_KEY]
    if vargin[T2_STAR_KEY]:
        print('Calculating T2_star')
        scan.generate_t2_star_map()

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    for tissue in scan.tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    return scan


def add_tissues(parser):
    for tissue in knee.SUPPORTED_TISSUES:
        parser.add_argument('-%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                            help='analyze %s' % tissue.FULL_NAME)


def parse_args():
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='pipeline',
                                     description='Tool for segmenting MRI knee volumes',
                                     epilog='Either `-d` or `-l` must be specified. '
                                            'If both are given, `-d` will be used')
    parser.add_argument('--%s' % DEBUG_KEY, action='store_const', const=True, default=False, help='debug')

    # Dicom and results paths
    parser.add_argument('-d', '--%s' % DICOM_KEY, metavar='D', type=str, default=None, nargs='?',
                        help='path to directory storing dicom files')
    parser.add_argument('-l', '--%s' % LOAD_KEY, metavar='L', type=str, default=None, nargs='?',
                        help='path to data directory to load from')
    parser.add_argument('-s', '--%s' % SAVE_KEY, metavar='S', type=str, default=None, nargs='?',
                        help='path to data directory to save to. Default: L/D')

    supported_format_names = [data_format.name for data_format in SUPPORTED_FORMATS]
    parser.add_argument('-df', '--%s' % DATA_FORMAT_KEY, metavar='F', type=str,
                        default=defaults.DEFAULT_OUTPUT_IMAGE_DATA_FORMAT.name, nargs='?',
                        choices=supported_format_names,
                        help='data format to store information in %s. Default: %s' % (str(supported_format_names),
                                                                                      defaults.DEFAULT_OUTPUT_IMAGE_DATA_FORMAT.name))

    parser.add_argument('-%s' % GPU_KEY, metavar='G', type=str, default=None, nargs='?', help='gpu id. Default: None')

    subparsers = parser.add_subparsers(help='sub-command help', dest=SCAN_KEY)

    # qDESS parser
    parser_qdess = subparsers.add_parser(qDESS_SCAN_KEY, help='analyze qDESS sequence')
    parser_qdess.add_argument('-%s' % T2_KEY, action='store_const', default=False, const=True, help='compute T2 map')

    add_tissues(parser_qdess)

    subparsers_qdess = parser_qdess.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    segmentation_parser_qdess = add_segmentation_subparser(subparsers_qdess)

    # add additional fields to base segmentation
    segmentation_parser_qdess.add_argument('-%s' % USE_RMS_KEY, action='store_const', default=False, const=True,
                                           help='use root mean square (rms) of two echos for segmentation')

    parser_qdess.set_defaults(func=handle_qdess)

    # Cubequant parser
    parser_cubequant = subparsers.add_parser(CUBEQUANT_SCAN_KEYS[0],
                                             help='analyze cubequant sequence',
                                             aliases=CUBEQUANT_SCAN_KEYS[1:])
    parser_cubequant.add_argument('-%s' % T1_RHO_Key,
                                  action='store_const',
                                  default=False,
                                  const=True,
                                  help='do t1-rho analysis')
    add_tissues(parser_cubequant)

    subparsers_cubequant = parser_cubequant.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    add_interregister_subparser(subparsers_cubequant)
    parser_cubequant.set_defaults(func=handle_cubequant)

    # Cones parser
    parser_cones = subparsers.add_parser(CONES_KEY, help='analyze cones sequence')
    parser_cones.add_argument('-%s' % T2_STAR_KEY, action='store_const', default=False, const=True,
                              help='do t2* analysis')
    add_tissues(parser_cones)

    subparsers_cones = parser_cones.add_subparsers(help='sub-command help', dest=ACTION_KEY)
    add_interregister_subparser(subparsers_cones)
    parser_cones.set_defaults(func=handle_cones)

    # MSK knee parser
    knee.knee_parser(subparsers)

    start_time = time.time()
    args = parser.parse_args()
    vargin = vars(args)

    if vargin[DEBUG_KEY]:
        fc.set_debug()

    gpu = vargin[GPU_KEY]

    if fc.DEBUG:
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
        save_path = load_path if load_path else dicom_path
        vargin[SAVE_KEY] = save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tissues = parse_tissues(vargin)
    vargin['tissues'] = tissues
    vargin[DATA_FORMAT_KEY] = ImageDataFormat[vargin[DATA_FORMAT_KEY]]

    # Call func for specific scan (qDESS, cubequant, cones, etc)
    scan_or_tissues = args.func(vargin)

    print('Time Elapsed: %0.2f seconds' % (time.time() - start_time))


if __name__ == '__main__':
    raise DeprecationWarning('This file is deprecated. Use dosma.py instead')
    parse_args()
