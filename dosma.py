"""
Main file for scan pipeline - handle argparse
"""

import argparse
import inspect
import os
import time

import defaults
import file_constants as fc
from data_io.format_io import ImageDataFormat, SUPPORTED_FORMATS
from models.get_model import SUPPORTED_MODELS
from models.get_model import get_model
from models.model import SegModel
from msk import knee
from scan_sequences.cube_quant import CubeQuant
from scan_sequences.qdess import QDess
from tissues.tissue import Tissue
from utils.quant_vals import QuantitativeValues as QV
from data_io.fig_format import SUPPORTED_VISUALIZATION_FORMATS


SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]

DEBUG_KEY = 'debug'

DICOM_KEY = 'dicom'
SAVE_KEY = 'save'
LOAD_KEY = 'load'
DATA_FORMAT_KEY = 'format'
VISUALIZATION_FORMAT_KEY = 'vis_format'
GPU_KEY = 'gpu'

SCAN_KEY = 'scan'
SCAN_ACTION_KEY = 'scan_action'

SEGMENTATION_MODEL_KEY = 'model'
SEGMENTATION_WEIGHTS_DIR_KEY = 'weights_dir'
SEGMENTATION_BATCH_SIZE_KEY = 'batch_size'

TISSUES_KEY = 'tissues'

SUPPORTED_SCAN_TYPES = [QDess, CubeQuant]
BASIC_TYPES = [bool, str, float, int, list, tuple]


def get_nargs_for_basic_type(base_type):
    if base_type in [str, float, int]:
        return 1
    elif base_type in [list, tuple]:
        return '+'

def add_tissues(parser):
    for tissue in knee.SUPPORTED_TISSUES:
        parser.add_argument('--%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                            help='analyze %s' % tissue.FULL_NAME)


def parse_tissues(vargin):
    tissues = []
    for tissue in knee.SUPPORTED_TISSUES:
        t = tissue()
        if t.STR_ID in vargin.keys() and vargin[t.STR_ID] and t.STR_ID not in [x.STR_ID for x in tissues]:
            load_path = vargin[LOAD_KEY]
            if load_path:
                t.load_data(load_path)

            tissues.append(t)

    # if no tissues are specified, do computation for all supported tissues
    if len(tissues) == 0:
        print('No tissues specified, computing for all supported tissues...')
        tissues = []
        for tissue in knee.SUPPORTED_TISSUES:
            t = tissue()
            if t.STR_ID not in [x.STR_ID for x in tissues]:
                load_path = vargin[LOAD_KEY]
                if load_path:
                    t.load_data(load_path)

                tissues.append(t)

    analysis_str = 'Tissue(s): '
    for tissue in tissues:
        analysis_str += '%s, ' % tissue.FULL_NAME

    print(analysis_str)

    return tissues


def add_segmentation_subparser(parser):
    parser.add_argument('--%s' % SEGMENTATION_WEIGHTS_DIR_KEY, type=str, nargs=1,
                        required=True,
                        help='path to directory with weights')
    parser.add_argument('--%s' % SEGMENTATION_MODEL_KEY, choices=SUPPORTED_MODELS, nargs='?', default='unet2d',
                        help='Model to use for segmentation. Choices: {%s}' % 'unet2d')
    parser.add_argument('--%s' % SEGMENTATION_BATCH_SIZE_KEY, metavar='B', type=int,
                        default=defaults.DEFAULT_BATCH_SIZE, nargs='?',
                        help='batch size for inference. Default: %d' % defaults.DEFAULT_BATCH_SIZE)

    return parser


def handle_segmentation(vargin, scan, tissue):
    segment_weights_path = vargin[SEGMENTATION_WEIGHTS_DIR_KEY][0]
    tissue.find_weights(segment_weights_path)

    # Load model
    dims = scan.get_dimensions()
    input_shape = (dims[0], dims[1], 1)
    model = get_model(vargin[SEGMENTATION_MODEL_KEY],
                      input_shape=input_shape,
                      weights_path=tissue.weights_filepath)
    model.batch_size = vargin[SEGMENTATION_BATCH_SIZE_KEY]

    return model


CUSTOM_TYPE_TO_HANDLE_DICT = {SegModel: handle_segmentation}


def add_custom_argument(parser, param_type):
    # handle all custom arguments except tissues
    has_custom_argument = False
    if param_type is SegModel:
        add_segmentation_subparser(parser)
        has_custom_argument = True

    return has_custom_argument


def add_base_argument(parser: argparse.ArgumentParser, param_name, param_type, param_default, param_help,
                      additional_param_names: list = []):
    assert param_type in BASIC_TYPES, "type %s not in BASIC_TYPES" % param_type

    # add default value to param help
    has_default = param_default is not inspect._empty
    if has_default:
        param_help = '%s. Default: %s' % (param_help, param_default)

    if additional_param_names:
        param_names = ['--%s' % n for n in additional_param_names]
    else:
        param_names = []

    param_names.append('--%s' % param_name)

    if param_type is bool:
        if not has_default:
            raise ValueError('All boolean parameters must have a default value.')

        parser.add_argument(*param_names, action='store_%s' % (str(not param_default).lower()),
                            dest=param_name,
                            help=param_help)
        return

    # all other values with default have this parameter
    nargs_no_default = get_nargs_for_basic_type(param_type)
    nargs = '?' if has_default else nargs_no_default

    parser.add_argument(*param_names, nargs=nargs,
                        default=param_default if has_default else None,
                        dest=param_name,
                        help=param_help,
                        required=not has_default)

def parse_basic_type(val, param_type):
    if type(val) is param_type:
        return val

    if param_type in [list, tuple]:
        return param_type(val)

    nargs = get_nargs_for_basic_type(param_type)
    if type(val) is list and nargs == 1:
        return val[0]
    return param_type(val) if val else val

def add_scans(dosma_subparser):
    for scan in SUPPORTED_SCAN_TYPES:
        supported_actions = scan.cmd_line_actions()

        # skip scans that are not supported on the command line
        if len(supported_actions) == 0:
            continue
        scan_name = scan.NAME
        scan_parser = dosma_subparser.add_parser(scan.NAME, help='analyze %s sequence' % scan_name)
        add_tissues(scan_parser)

        if not len(supported_actions):
            raise ValueError('No supported actions for scan %s' % scan_name)

        scan_subparser = scan_parser.add_subparsers(description='%s subcommands' % scan.NAME,
                                                    dest=SCAN_ACTION_KEY)

        for action, action_wrapper in supported_actions:
            func_signature = inspect.signature(action)
            func_name = action_wrapper.name
            aliases = action_wrapper.aliases
            action_parser = scan_subparser.add_parser(func_name, aliases=aliases, help=action_wrapper.help)

            parameters = func_signature.parameters
            for param_name in parameters.keys():
                param = parameters[param_name]
                param_type = param.annotation
                param_default = param.default

                if param_name == 'self' or param_type is Tissue:
                    continue

                param_help = action_wrapper.get_param_help(param_name)
                alternative_param_names = action_wrapper.get_alternative_param_names(param_name)

                if param_type is inspect._empty:
                    raise ValueError(
                        'scan %s, action %s, param %s does not have an annotation. Use pytying in the method declaration' % (
                        scan.NAME, func_name, param_name))

                # see if the type is a custom type, if not handle it as a basic type
                is_custom_arg = add_custom_argument(action_parser, param_type)
                if is_custom_arg:
                    continue

                add_base_argument(action_parser, param_name, param_type, param_default,
                                  param_help=param_help,
                                  additional_param_names=alternative_param_names)

        scan_parser.set_defaults(func=handle_scan)


def handle_scan(vargin):
    scan_name = vargin[SCAN_KEY]
    print('Analyzing %s...' % scan_name)
    scan = None

    for p_scan in SUPPORTED_SCAN_TYPES:
        if p_scan.NAME == scan_name:
            scan = p_scan
            break

    scan = scan(dicom_path=vargin[DICOM_KEY], load_path=vargin[LOAD_KEY])
    tissues = vargin['tissues']
    scan_action = vargin[SCAN_ACTION_KEY]

    p_action = None
    for action, action_wrapper in scan.cmd_line_actions():
        if scan_action == action_wrapper.name or scan_action in action_wrapper.aliases:
            p_action = action
            break

    # search for name in the cmd_line actions
    action = p_action

    func_signature = inspect.signature(action)
    parameters = func_signature.parameters
    for tissue in tissues:
        param_dict = {}
        for param_name in parameters.keys():
            param = parameters[param_name]
            param_type = param.annotation

            if param_name == 'self':
                continue

            if param_type is Tissue:
                param_dict['tissue'] = tissue
                continue

            if param_type in CUSTOM_TYPE_TO_HANDLE_DICT:
                param_dict[param_name] = CUSTOM_TYPE_TO_HANDLE_DICT[param_type](vargin, scan, tissue)
            else:
                param_dict[param_name] = parse_basic_type(vargin[param_name], param_type)

        scan.__getattribute__(action.__name__)(**param_dict)

    scan.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])
    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    return scan


def parse_args(f_input=None):
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='DOSMA',
                                     description='A deep-learning powered open source MRI analysis pipeline',
                                     epilog='Either `--d` or `---l` must be specified. '
                                            'If both are given, `-d` will be used')
    parser.add_argument('--%s' % DEBUG_KEY, action='store_true', help='use debug mode')

    # Dicom and results paths
    parser.add_argument('--d', '--%s' % DICOM_KEY, metavar='D', type=str, default=None, nargs='?',
                        dest=DICOM_KEY,
                        help='path to directory storing dicom files')
    parser.add_argument('--l', '--%s' % LOAD_KEY, metavar='L', type=str, default=None, nargs='?',
                        dest=LOAD_KEY,
                        help='path to data directory to load from')
    parser.add_argument('--s', '--%s' % SAVE_KEY, metavar='S', type=str, default=None, nargs='?',
                        dest=SAVE_KEY,
                        help='path to data directory to save to. Default: L/D')

    supported_format_names = [data_format.name for data_format in SUPPORTED_FORMATS]
    parser.add_argument('--df', '--%s' % DATA_FORMAT_KEY, metavar='F', type=str,
                        dest=DATA_FORMAT_KEY,
                        default=defaults.DEFAULT_OUTPUT_IMAGE_DATA_FORMAT.name, nargs='?',
                        choices=supported_format_names,
                        help='data format to store information in %s. Default: %s' % (str(supported_format_names),
                                                                                      defaults.DEFAULT_OUTPUT_IMAGE_DATA_FORMAT.name))
    parser.add_argument('--vf', '--%s' % VISUALIZATION_FORMAT_KEY, metavar='V', type=str,
                        dest=VISUALIZATION_FORMAT_KEY,
                        default=defaults.DEFAULT_FIG_FORMAT,
                        nargs='?',
                        choices=SUPPORTED_VISUALIZATION_FORMATS,
                        help='visualization format %s. Default: %s' % (str(tuple(SUPPORTED_VISUALIZATION_FORMATS)),
                                                                       defaults.DEFAULT_FIG_FORMAT))

    parser.add_argument('--%s' % GPU_KEY, metavar='G', type=str, default=None, nargs='?',
                        dest=GPU_KEY,
                        help='gpu id. Default: None')

    subparsers = parser.add_subparsers(help='sub-command help', dest=SCAN_KEY)
    add_scans(subparsers)

    # MSK knee parser
    knee.knee_parser(subparsers)

    start_time = time.time()
    if f_input:
        args = parser.parse_args(f_input)
    else:
        args = parser.parse_args()

    vargin = vars(args)

    if vargin[DEBUG_KEY]:
        fc.set_debug()

    gpu = vargin[GPU_KEY]

    if fc.DEBUG:
        print(vargin)

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    dicom_path = vargin[DICOM_KEY]
    load_path = vargin[LOAD_KEY]

    if not dicom_path and not load_path:
        raise ValueError('Must provide path to dicoms or path to load data from')

    save_path = vargin[SAVE_KEY]
    if not save_path:
        save_path = load_path if load_path else '%s/data' % dicom_path
        vargin[SAVE_KEY] = save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tissues = parse_tissues(vargin)
    vargin['tissues'] = tissues
    vargin[DATA_FORMAT_KEY] = ImageDataFormat[vargin[DATA_FORMAT_KEY]]
    defaults.DEFAULT_FIG_FORMAT = vargin[VISUALIZATION_FORMAT_KEY]

    args.func(vargin)

    time_elapsed = (time.time() - start_time)
    print('Time Elapsed: %0.2f seconds' % (time.time() - start_time))

    return time_elapsed


if __name__ == '__main__':
    parse_args()
