"""Initialize and parse command line arguments for DOSMA.

This module is the entry point for executing DOSMA from the command line. The DOSMA library is critical for processing.

To use this file, it must be run as a module from the parent directory::

    $ python -m dosma/cli ...

Example:
    Run T2 fitting on Subject 01, Series 007, a quantitative DESS (qDESS) scan, for femoral cartilage::

        $ python -m dosma/cli --dicom subject01/dicoms/007/ --save subject01/data/ qdess --fc generate_t2_map

Hint:
    Run `python -m dosma/cli --help` for a detailed description of different command line arguments.
"""

import argparse
import ast
import inspect
import os
import time
from collections import defaultdict
from typing import Sequence

from dosma import file_constants as fc
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel
from dosma.models.util import SUPPORTED_MODELS
from dosma.models.util import get_model, model_from_config
from dosma.msk import knee
from dosma.scan_sequences.cones import Cones
from dosma.scan_sequences.cube_quant import CubeQuant
from dosma.scan_sequences.mapss import Mapss
from dosma.scan_sequences.qdess import QDess
from dosma.scan_sequences.scans import ScanSequence
from dosma.quant_vals import QuantitativeValueType as QV
from dosma.tissues.tissue import Tissue
from dosma.utils import io_utils

import logging

SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]

DEBUG_KEY = 'debug'

DICOM_KEY = 'dicom'
SAVE_KEY = 'save'
LOAD_KEY = 'load'
IGNORE_EXT_KEY = 'ignore_ext'
SPLIT_BY_KEY = 'split_by'

GPU_KEY = 'gpu'

SCAN_KEY = 'scan'
SCAN_ACTION_KEY = 'scan_action'

SEGMENTATION_MODEL_KEY = 'model'
SEGMENTATION_CONFIG_KEY = 'config'
SEGMENTATION_WEIGHTS_DIR_KEY = 'weights_dir'
SEGMENTATION_BATCH_SIZE_KEY = 'batch_size'

TISSUES_KEY = 'tissues'

SUPPORTED_SCAN_TYPES = [Cones, CubeQuant, Mapss, QDess]
BASIC_TYPES = [bool, str, float, int, list, tuple]


def get_nargs_for_basic_type(base_type: type):
    if base_type in [str, float, int]:
        return 1
    elif base_type in [list, tuple]:
        return '+'


def add_tissues(parser: argparse.ArgumentParser):
    for tissue in knee.SUPPORTED_TISSUES:
        parser.add_argument('--%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                            help='analyze %s' % tissue.FULL_NAME)


def parse_tissues(vargin: dict):
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
        logging.info('No tissues specified, computing for all supported tissues...')
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

    logging.info(analysis_str)

    return tissues


def add_segmentation_subparser(parser):
    parser.add_argument('--%s' % SEGMENTATION_WEIGHTS_DIR_KEY, type=str, nargs=1,
                        required=True,
                        help='path to directory with weights')
    parser.add_argument('--%s' % SEGMENTATION_MODEL_KEY, choices=SUPPORTED_MODELS, nargs='?', default=None,
                        help='built-in model to use for segmentation. Choices: %s' % SUPPORTED_MODELS)
    parser.add_argument('--%s' % SEGMENTATION_CONFIG_KEY, type=str, default=None, help='config file for non-built-in model')
    parser.add_argument('--%s' % SEGMENTATION_BATCH_SIZE_KEY, metavar='B', type=int,
                        default=preferences.segmentation_batch_size, nargs='?',
                        help='batch size for inference. Default: %d' % preferences.segmentation_batch_size)

    return parser


def handle_segmentation(vargin, scan: ScanSequence, tissue: Tissue):
    if not vargin[SEGMENTATION_MODEL_KEY] and not vargin[SEGMENTATION_CONFIG_KEY]:
        raise ValueError("Either `--{}` or `--{}` must be specified".format(
            SEGMENTATION_MODEL_KEY, SEGMENTATION_CONFIG_KEY,
        ))

    segment_weights_path = vargin[SEGMENTATION_WEIGHTS_DIR_KEY][0]
    if isinstance(tissue, Sequence):
        weights = [t.find_weights(segment_weights_path) for t in tissue]
        assert all(weights_file == weights[0] for weights_file in weights)
        weights_path = weights[0]
    else:
        weights_path = tissue.find_weights(segment_weights_path)

    # Load model
    dims = scan.get_dimensions()
    # TODO: Input shape should be determined by combination of model + scan.
    # Currently fixed in 2D plane
    input_shape = (dims[0], dims[1], 1)
    if vargin[SEGMENTATION_MODEL_KEY]:
        # Use built-in model
        model = get_model(
            vargin[SEGMENTATION_MODEL_KEY],
            input_shape=input_shape,
            weights_path=weights_path,
        )
    else:
        # Use config
        model = model_from_config(
            vargin[SEGMENTATION_CONFIG_KEY],
            weights_dir=segment_weights_path,
            input_shape=input_shape
        )
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


def _find_tissue_groups(vargin, tissues: Sequence[Tissue]):
    """Finds groups of tissues that can be segmented together.

    Some segmentation models have multiple weight files for different tissues.
    Many models have one weight file for multiple tissues (i.e. multi-class segmentation).

    This function matches tissues with their corresponding weight file.
    If multiple tissues share a single weight file, they will map to the same weight file,
    allowing multiple tissues to be segmented simultaneously.

    This is a temporary fix for segmenting multiple classes.
    It should not be extended or used as precedence for future development.
    """
    if not isinstance(tissues, Sequence):
        assert isinstance(tissues, Tissue)
        tissues = [tissues]

    weights_dir = vargin[SEGMENTATION_WEIGHTS_DIR_KEY][0]
    weights_to_tissues = defaultdict(list)
    for tissue in tissues:
        weights_to_tissues[tissue.find_weights(weights_dir)].append(tissue)

    return weights_to_tissues


def _build_params(vargin, scan, parameters, tissue):
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
            param_dict[param_name] = CUSTOM_TYPE_TO_HANDLE_DICT[param_type](
                vargin, scan, tissue
            )
        else:
            param_dict[param_name] = parse_basic_type(vargin[param_name], param_type)
    return param_dict


def handle_scan(vargin):

    scan_name = vargin[SCAN_KEY]
    logging.info("Analyzing {}...".format(scan_name))
    scan = None

    for p_scan in SUPPORTED_SCAN_TYPES:
        if p_scan.NAME == scan_name:
            scan = p_scan
            break

    scan = scan(
        dicom_path=vargin[DICOM_KEY],
        load_path=vargin[LOAD_KEY],
        ignore_ext=vargin[IGNORE_EXT_KEY],
        split_by=vargin[SPLIT_BY_KEY] if vargin[SPLIT_BY_KEY] else scan.__DEFAULT_SPLIT_BY__,
    )

    tissues = vargin['tissues']
    scan_action = scan_action_str = vargin[SCAN_ACTION_KEY]

    p_action = None
    for action, action_wrapper in scan.cmd_line_actions():
        if scan_action == action_wrapper.name or scan_action in action_wrapper.aliases:
            p_action = action
            break

    # search for name in the cmd_line actions
    action = p_action

    if action is None:
        scan.save_data(vargin[SAVE_KEY], data_format=preferences.image_data_format)
        return

    func_signature = inspect.signature(action)
    parameters = func_signature.parameters
    if scan_action_str == "segment":
        weights_to_tissues = _find_tissue_groups(vargin, tissues)
        for weights_file, seg_tissues in weights_to_tissues.items():
            if len(seg_tissues) == 1:
                seg_tissues = seg_tissues[0]
            param_dict = _build_params(vargin, scan, parameters, seg_tissues)
            scan.__getattribute__(action.__name__)(**param_dict)
    else:
        for tissue in tissues:
            param_dict = _build_params(vargin, scan, parameters, tissue)
            scan.__getattribute__(action.__name__)(**param_dict)

    scan.save_data(vargin[SAVE_KEY], data_format=preferences.image_data_format)
    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=preferences.image_data_format)

    return scan


def parse_dicom_tag_splitby(vargin_str):
    if not vargin_str:
        return vargin_str

    try:
        parsed_str = ast.literal_eval(vargin_str)
        return parsed_str
    except:
        return vargin_str


def parse_args(f_input=None):
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(prog='DOSMA',
                                     description='A deep-learning powered open source MRI analysis pipeline',
                                     epilog='Either `--dicom` or `--load` must be specified. '
                                            'If both are given, `--dicom` will be used')
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
    parser.add_argument('--%s' % IGNORE_EXT_KEY, action='store_true', default=False,
                        dest=IGNORE_EXT_KEY,
                        help='ignore .dcm extension when loading dicoms. Default: False')
    parser.add_argument('--%s' % SPLIT_BY_KEY, metavar='G', type=str, default=None, nargs='?',
                        dest=SPLIT_BY_KEY,
                        help='override dicom tag to split volumes by (eg. `EchoNumbers`)')

    parser.add_argument('--%s' % GPU_KEY, metavar='G', type=str, default=None, nargs='?',
                        dest=GPU_KEY,
                        help='gpu id. Default: None')

    # Add preferences
    preferences_flags = preferences.cmd_line_flags()
    for flag in preferences_flags.keys():
        argparse_kwargs = preferences_flags[flag]
        argparse_kwargs['dest'] = flag
        aliases = argparse_kwargs.pop('aliases', None)
        name = argparse_kwargs.pop('name', None)
        parser.add_argument(*aliases, **argparse_kwargs)

    subparsers = parser.add_subparsers(help='sub-command help', dest=SCAN_KEY)
    add_scans(subparsers)

    # MSK knee parser
    knee.knee_parser(subparsers)

    start_time = time.time()
    if f_input:
        args = parser.parse_args(f_input)
    else:
        args = parser.parse_args()

        # Only initialize logger if called from command line.
        # If UI is using it, the logger should be initialized by the UI.
        io_utils.init_logger(fc.LOG_FILE_PATH, args.debug)

    vargin = vars(args)

    if vargin[DEBUG_KEY]:
        fc.set_debug()

    gpu = vargin[GPU_KEY]

    logging.debug(vargin)

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # parse and update preferences
    for flag in preferences_flags.keys():
        preferences.set(flag, vargin[flag])

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

    vargin[SPLIT_BY_KEY] = parse_dicom_tag_splitby(vargin[SPLIT_BY_KEY])

    args.func(vargin)

    time_elapsed = (time.time() - start_time)
    logging.info("Time Elapsed: {:.2f} seconds".format(time.time() - start_time))

    return time_elapsed


if __name__ == '__main__':
    parse_args()
