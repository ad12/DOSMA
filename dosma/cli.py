"""Initialize and parse command line arguments for DOSMA.

This module is the entry point for executing DOSMA from the command line.
The DOSMA library is critical for processing.

To use this file, it must be run as a module from the parent directory::

    $ python -m dosma/cli ...

Examples:
    Run T2 fitting on Subject 01, Series 007, a quantitative DESS (qDESS) scan,
    for femoral cartilage::

    python -m dosma/cli --dicom subject01/dicoms/007/ --save subject01/data/ \
        qdess --fc generate_t2_map

Hint:
    Run ``python -m dosma/cli --help`` for a detailed description of different
    command line arguments.
"""

import argparse
import ast
import functools
import inspect
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import Sequence

from dosma.core.io.format_io import ImageDataFormat
from dosma.core.quant_vals import QuantitativeValueType as QV
from dosma.defaults import preferences
from dosma.models.seg_model import SegModel
from dosma.models.util import SUPPORTED_MODELS, get_model, model_from_config
from dosma.msk import knee
from dosma.scan_sequences.mri.cones import Cones
from dosma.scan_sequences.mri.cube_quant import CubeQuant
from dosma.scan_sequences.mri.mapss import Mapss
from dosma.scan_sequences.mri.qdess import QDess
from dosma.scan_sequences.scans import ScanSequence
from dosma.tissues.tissue import Tissue
from dosma.utils import env
from dosma.utils.logger import setup_logger

SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]

DEBUG_KEY = "debug"

DICOM_KEY = "dicom"
SAVE_KEY = "save"
LOAD_KEY = "load"
IGNORE_EXT_KEY = "ignore_ext"
SPLIT_BY_KEY = "split_by"

GPU_KEY = "gpu"
NUM_WORKERS_KEY = "num-workers"

SCAN_KEY = "scan"
SCAN_ACTION_KEY = "scan_action"

SEGMENTATION_MODEL_KEY = "model"
SEGMENTATION_CONFIG_KEY = "config"
SEGMENTATION_WEIGHTS_DIR_KEY = "weights_dir"
SEGMENTATION_BATCH_SIZE_KEY = "batch_size"

TISSUES_KEY = "tissues"

SUPPORTED_SCAN_TYPES = [Cones, CubeQuant, Mapss, QDess]
BASIC_TYPES = [bool, str, float, int, list, tuple]

_logger = logging.getLogger(__name__)


class CommandLineScanContainer:
    def __init__(
        self,
        scan_type: type,
        dicom_path,
        load_path,
        ignore_ext: bool = False,
        group_by=None,
        num_workers=0,
        **kwargs,
    ):
        """The class for command-line handling around :class:`ScanSequence`.

        The command line interface for :class:`ScanSequence` data is highly structured,
        particularly in data saving and loading to support easy and accurate command-line
        compatibility.

        This class overloads some standard functionality in :class:`ScanSequence`
        (:func:`save`, :func:`load`). When methods are not implemented, this class provides
        access directly to attributes/methods of the underlying scan instantiation.
        For example, if ``scan_type=QDess``, the following will call
        :func:`QDess.generate_t2_map`:

        >>> cli_scan = CommandLineScanContainer(QDess, dicom_path="/path/to/qdess/scan")
        >>> cli_scan.generate_t2_map(...)  # this calls cli_scan.scan.generate_t2_map

        Data is loaded either from the ``dicom_path`` or the ``load_path``. If both are specified,
        the data is loaded from the ``dicom_path``.

        Args:
            scan_type (type): A scan type. Should be subclass of `ScanSequence`.
            dicom_path (str): The dicom path. This value can be ``None``, but must
                be explicitly set.
            load_path (str): The load path. This value can be ``None``, but must be
                explicitly set.
            ignore_ext (bool, optional): If ``True``, ignore extensions when loading
                dicom data. See :func:`DicomReader.load` for details.
            group_by (optional): The value(s) to group dicoms by. See :func:`DicomReader.load`
                for details.
            num_workers (int, optional): Number of works for loading scan.

        Attributes:
            scan_type (type): The scan type to instantiate.
            scan (ScanSequence): The instantiated scan.
            generic_args (Dict[str, Any]): Generic duck typed parameter names and values.
                If parameters with this name are part of the method signature, they will
                automatically be set to the values in this dictionary. Keys include:
                    * "num_workers": Number of cpu workers to use.
                    * "max_workers": Alias for "num_workers" in some functions
                    * "verbose": Verbosity
                    * "show_pbar": Show progress bar.

        Raises:
            NotADirectoryError: If ``dicom_path`` is not a path to a directory.
        """
        self.scan_type = scan_type

        if (dicom_path is not None) and (not os.path.isdir(dicom_path)):
            if load_path is not None:
                warnings.warn(
                    "Dicom_path {} not found. Will load data from {}".format(dicom_path, load_path)
                )
            else:
                raise NotADirectoryError("{} is not a directory".format(dicom_path))

        # Only use dicoms if the path exists and path contains files.
        is_dicom_available = (dicom_path is not None) and (os.path.isdir(dicom_path))

        # If dicom_path is specified and exists, assume user wants to start from scratch with the
        # dicoms. load_path is ignored.
        group_by = group_by if group_by is not None else scan_type.__DEFAULT_SPLIT_BY__
        if is_dicom_available:
            scan = scan_type.from_dicom(
                dicom_path, group_by=group_by, ignore_ext=ignore_ext, num_workers=num_workers
            )
        else:
            scan = self.load(load_path, num_workers=num_workers)

        self.scan = scan
        self.generic_args = {
            "num_workers": num_workers,
            "max_workers": num_workers,
            "verbose": True,
            "show_pbar": True,
        }

    def __getattr__(self, name):
        attr = getattr(self.scan, name)
        if callable(attr):
            params = inspect.signature(attr).parameters
            params = params.keys() & self.generic_args.keys()
            kwargs = {k: self.generic_args[k] for k in params}
            if len(kwargs):
                attr = functools.partial(attr, **kwargs)
        return attr

    def load(self, path: str, num_workers: int = 0):
        """Command line interface loading scan data.

        ``self.scan_type`` must be set before calling this function.

        Args
            path (str): Path to pickle file or directory where data is stored.
            num_workers (int, optional): Number of workers to use to load data.

        Returns:
            ScanSequence: Scan of type ``self.scan_type``.

        Raises:
            ValueError: If path to load data from cannot be determined.

        Examples:
            >>> cli_scan.load("/path/to/pickle/file")  # load data from pickle file
            >>> cli_scan.load("/path/to/directory")  # load data from directory
        """
        scan_type = self.scan_type

        file_path = None
        if os.path.isfile(path):
            file_path = path
        elif os.path.isdir(path) and scan_type.NAME:
            fname = f"{scan_type.NAME}.data"
            _paths = (
                os.path.join(path, fname),
                os.path.join(self._save_dir(path, create_dir=False), fname),
            )
            for _path in _paths:
                if os.path.isfile(_path):
                    file_path = _path
                    break
        if file_path is None:
            raise ValueError(f"Cannot load {scan_type.__name__} data from path '{path}'")

        return scan_type.load(file_path, num_workers)

    def _save_dir(self, dir_path: str, create_dir: bool = True):
        """Returns directory path specific to this scan.

        Formatted as '`base_load_dirpath`/`scan.NAME`'.

        Args:
            dir_path (str): Directory path where all data is stored.
            create_dir (`bool`, optional): If `True`, creates directory if it doesn't exist.

        Returns:
            str: Data directory path for this scan.
        """
        scan_type = self.scan_type
        folder_id = scan_type.NAME

        name_len = len(folder_id) + 2  # buffer
        if scan_type.NAME not in dir_path[-name_len:]:
            scan_dirpath = os.path.join(dir_path, folder_id)
        else:
            scan_dirpath = dir_path

        # scan_dirpath = os.path.join(scan_dirpath, folder_id)

        if create_dir:
            os.makedirs(scan_dirpath, exist_ok=True)

        return scan_dirpath

    def save(
        self,
        path: str,
        save_custom: bool = True,
        image_data_format: ImageDataFormat = None,
        num_workers: int = 0,
    ):
        path = self._save_dir(path, create_dir=True)
        return self.scan.save(path, save_custom, image_data_format, num_workers)


def get_nargs_for_basic_type(base_type: type):
    if base_type in [str, float, int]:
        return 1
    elif base_type in [list, tuple]:
        return "+"


def add_tissues(parser: argparse.ArgumentParser):
    for tissue in knee.SUPPORTED_TISSUES:
        parser.add_argument(
            "--%s" % tissue.STR_ID,
            action="store_const",
            default=False,
            const=True,
            help="analyze %s" % tissue.FULL_NAME,
        )


def parse_tissues(vargin: dict):
    tissues = []
    for tissue in knee.SUPPORTED_TISSUES:
        t = tissue()
        if (
            t.STR_ID in vargin.keys()
            and vargin[t.STR_ID]
            and t.STR_ID not in [x.STR_ID for x in tissues]
        ):
            load_path = vargin[LOAD_KEY]
            if load_path:
                t.load_data(load_path)

            tissues.append(t)

    # if no tissues are specified, do computation for all supported tissues
    if len(tissues) == 0:
        _logger.info("No tissues specified, computing for all supported tissues...")
        tissues = []
        for tissue in knee.SUPPORTED_TISSUES:
            t = tissue()
            if t.STR_ID not in [x.STR_ID for x in tissues]:
                load_path = vargin[LOAD_KEY]
                if load_path:
                    t.load_data(load_path)

                tissues.append(t)

    analysis_str = "Tissue(s): "
    for tissue in tissues:
        analysis_str += "%s, " % tissue.FULL_NAME

    _logger.info(analysis_str)

    return tissues


def add_segmentation_subparser(parser):
    parser.add_argument(
        "--%s" % SEGMENTATION_WEIGHTS_DIR_KEY,
        type=str,
        nargs=1,
        required=True,
        help="path to directory with weights",
    )
    parser.add_argument(
        "--%s" % SEGMENTATION_MODEL_KEY,
        choices=SUPPORTED_MODELS,
        nargs="?",
        default=None,
        help="built-in model to use for segmentation. Choices: %s" % SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--%s" % SEGMENTATION_CONFIG_KEY,
        type=str,
        default=None,
        help="config file for non-built-in model",
    )
    parser.add_argument(
        "--%s" % SEGMENTATION_BATCH_SIZE_KEY,
        metavar="B",
        type=int,
        default=preferences.segmentation_batch_size,
        nargs="?",
        help="batch size for inference. Default: %d" % preferences.segmentation_batch_size,
    )

    return parser


def handle_segmentation(vargin, scan: ScanSequence, tissue: Tissue):
    if not vargin[SEGMENTATION_MODEL_KEY] and not vargin[SEGMENTATION_CONFIG_KEY]:
        raise ValueError(
            "Either `--{}` or `--{}` must be specified".format(
                SEGMENTATION_MODEL_KEY, SEGMENTATION_CONFIG_KEY
            )
        )

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
            vargin[SEGMENTATION_MODEL_KEY], input_shape=input_shape, weights_path=weights_path
        )
    else:
        # Use config
        model = model_from_config(
            vargin[SEGMENTATION_CONFIG_KEY],
            weights_dir=segment_weights_path,
            input_shape=input_shape,
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


def add_base_argument(
    parser: argparse.ArgumentParser,
    param_name,
    param_type,
    param_default,
    param_help,
    additional_param_names: list = None,
):
    if additional_param_names is None:
        additional_param_names = []

    # TODO: Clean up this code block to properly do syntax parsing.
    try:
        if param_type not in BASIC_TYPES:
            param_type = extract_basic_type(param_type)
    except (AttributeError, TypeError):
        raise TypeError(
            "Parameter '{}' - type '{}' not in BASIC_TYPES".format(param_name, param_type)
        )

    # add default value to param help
    has_default = param_default is not inspect._empty
    if has_default:
        param_help = "%s. Default: %s" % (param_help, param_default)

    if additional_param_names:
        param_names = ["--%s" % n for n in additional_param_names]
    else:
        param_names = []

    param_names.append("--%s" % param_name)

    if param_type is bool:
        if not has_default:
            raise ValueError("All boolean parameters must have a default value.")

        parser.add_argument(
            *param_names,
            action="store_%s" % (str(not param_default).lower()),
            dest=param_name,
            help=param_help,
        )
        return

    # all other values with default have this parameter
    nargs_no_default = get_nargs_for_basic_type(param_type)
    nargs = "?" if has_default else nargs_no_default

    parser.add_argument(
        *param_names,
        nargs=nargs,
        default=param_default if has_default else None,
        dest=param_name,
        help=param_help,
        required=not has_default,
    )


def parse_basic_type(val, param_type):
    if param_type not in BASIC_TYPES:
        param_type = extract_basic_type(param_type)

    if type(val) is param_type:
        return val

    if param_type in [list, tuple]:
        return param_type(val)

    nargs = get_nargs_for_basic_type(param_type)
    if type(val) is list and nargs == 1:
        return val[0]
    return param_type(val) if val else val


def extract_basic_type(param_type):
    """Extracts basic types from ``typing`` aliases.

    Args:
        param_type (typing._GenericAlias): A generic alias
            (e.g. ``typing.Tuple``, ``typing.List``).

    Returns:
        type: The basic type.
    """
    try:
        # Python 3.5 / 3.6
        return param_type.__extra__
    except AttributeError:
        # Python 3.7/3.8/3.9
        return param_type.__origin__


def add_scans(dosma_subparser):
    for scan in SUPPORTED_SCAN_TYPES:
        supported_actions = scan.cmd_line_actions()

        # skip scans that are not supported on the command line
        if len(supported_actions) == 0:
            continue
        scan_name = scan.NAME
        scan_parser = dosma_subparser.add_parser(scan.NAME, help="analyze %s sequence" % scan_name)
        add_tissues(scan_parser)

        if not len(supported_actions):
            raise ValueError("No supported actions for scan %s" % scan_name)

        scan_subparser = scan_parser.add_subparsers(
            description="%s subcommands" % scan.NAME, dest=SCAN_ACTION_KEY
        )

        for action, action_wrapper in supported_actions:
            func_signature = inspect.signature(action)
            func_name = action_wrapper.name
            aliases = action_wrapper.aliases
            action_parser = scan_subparser.add_parser(
                func_name, aliases=aliases, help=action_wrapper.help
            )

            parameters = func_signature.parameters
            for param_name in parameters.keys():
                param = parameters[param_name]
                param_type = param.annotation
                param_default = param.default

                if param_name == "self" or param_type is Tissue:
                    continue

                param_help = action_wrapper.get_param_help(param_name)
                alternative_param_names = action_wrapper.get_alternative_param_names(param_name)

                if param_type is inspect._empty:
                    raise ValueError(
                        "scan %s, action %s, param %s does not have an annotation. "
                        "Use pytying in the method declaration" % (scan.NAME, func_name, param_name)
                    )

                # see if the type is a custom type, if not handle it as a basic type
                is_custom_arg = add_custom_argument(action_parser, param_type)
                if is_custom_arg:
                    continue

                add_base_argument(
                    action_parser,
                    param_name,
                    param_type,
                    param_default,
                    param_help=param_help,
                    additional_param_names=alternative_param_names,
                )

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


def _build_params(vargin, scan, parameters, tissue=None):
    param_dict = {}
    for param_name in parameters.keys():
        param = parameters[param_name]
        param_type = param.annotation

        if param_name == "self":
            continue

        if param_type is Tissue:
            assert tissue is not None
            param_dict["tissue"] = tissue
            continue

        if param_type in CUSTOM_TYPE_TO_HANDLE_DICT:
            param_dict[param_name] = CUSTOM_TYPE_TO_HANDLE_DICT[param_type](vargin, scan, tissue)
        else:
            param_dict[param_name] = parse_basic_type(vargin[param_name], param_type)
    return param_dict


def handle_scan(vargin):

    scan_name = vargin[SCAN_KEY]
    _logger.info("Analyzing {}...".format(scan_name))
    scan = None

    for p_scan in SUPPORTED_SCAN_TYPES:
        if p_scan.NAME == scan_name:
            scan = p_scan
            break

    scan = CommandLineScanContainer(
        scan,
        dicom_path=vargin[DICOM_KEY],
        load_path=vargin[LOAD_KEY],
        ignore_ext=vargin[IGNORE_EXT_KEY],
        split_by=vargin[SPLIT_BY_KEY] if vargin[SPLIT_BY_KEY] else scan.__DEFAULT_SPLIT_BY__,
        num_workers=vargin[NUM_WORKERS_KEY],
    )

    tissues = vargin["tissues"]
    scan_action = scan_action_str = vargin[SCAN_ACTION_KEY]

    p_action = None
    for action, action_wrapper in scan.cmd_line_actions():
        if scan_action == action_wrapper.name or scan_action in action_wrapper.aliases:
            p_action = action
            break

    # search for name in the cmd_line actions
    action = p_action

    if action is None:
        scan.save(vargin[SAVE_KEY], image_data_format=preferences.image_data_format)
        return

    func_signature = inspect.signature(action)
    parameters = func_signature.parameters
    if scan_action_str == "segment":
        weights_to_tissues = _find_tissue_groups(vargin, tissues)
        for _weights_file, seg_tissues in weights_to_tissues.items():
            if len(seg_tissues) == 1:
                seg_tissues = seg_tissues[0]
            param_dict = _build_params(vargin, scan, parameters, seg_tissues)
            getattr(scan, action.__name__)(**param_dict)
    else:
        if "tissue" in func_signature.parameters.keys():
            for tissue in tissues:
                param_dict = _build_params(vargin, scan, parameters, tissue)
                getattr(scan, action.__name__)(**param_dict)
        else:
            param_dict = _build_params(vargin, scan, parameters)
            getattr(scan, action.__name__)(**param_dict)

    scan.save(vargin[SAVE_KEY], image_data_format=preferences.image_data_format)
    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=preferences.image_data_format)

    return scan


def parse_dicom_tag_splitby(vargin_str):
    if not vargin_str:
        return vargin_str

    try:
        parsed_str = ast.literal_eval(vargin_str)
        return parsed_str
    except Exception:
        return vargin_str


def parse_args(f_input=None):
    """Parse arguments given through command line (argv)

    :raises ValueError if dicom path is not provided
    :raises NotADirectoryError if dicom path does not exist or is not a directory
    """
    parser = argparse.ArgumentParser(
        prog="DOSMA",
        description="A deep-learning powered open source MRI analysis pipeline",
        epilog="Either `--dicom` or `--load` must be specified. "
        "If both are given, `--dicom` will be used",
    )
    parser.add_argument("--%s" % DEBUG_KEY, action="store_true", help="use debug mode")

    # Dicom and results paths
    parser.add_argument(
        "--d",
        "--%s" % DICOM_KEY,
        metavar="D",
        type=str,
        default=None,
        nargs="?",
        dest=DICOM_KEY,
        help="path to directory storing dicom files",
    )
    parser.add_argument(
        "--l",
        "--%s" % LOAD_KEY,
        metavar="L",
        type=str,
        default=None,
        nargs="?",
        dest=LOAD_KEY,
        help="path to data directory to load from",
    )
    parser.add_argument(
        "--s",
        "--%s" % SAVE_KEY,
        metavar="S",
        type=str,
        default=None,
        nargs="?",
        dest=SAVE_KEY,
        help="path to data directory to save to. Default: L/D",
    )
    parser.add_argument(
        "--%s" % IGNORE_EXT_KEY,
        action="store_true",
        default=False,
        dest=IGNORE_EXT_KEY,
        help="ignore .dcm extension when loading dicoms. Default: False",
    )
    parser.add_argument(
        "--%s" % SPLIT_BY_KEY,
        metavar="G",
        type=str,
        default=None,
        nargs="?",
        dest=SPLIT_BY_KEY,
        help="override dicom tag to split volumes by (eg. `EchoNumbers`)",
    )

    parser.add_argument(
        "--%s" % GPU_KEY,
        metavar="G",
        type=str,
        default=None,
        nargs="?",
        dest=GPU_KEY,
        help="gpu id. Default: None",
    )

    parser.add_argument(
        "--%s" % NUM_WORKERS_KEY,
        metavar="G",
        type=int,
        default=0,
        dest=NUM_WORKERS_KEY,
        help="num cpu workers. Default: 0",
    )

    # Add preferences
    preferences_flags = preferences.cmd_line_flags()
    for flag in preferences_flags.keys():
        argparse_kwargs = preferences_flags[flag]
        argparse_kwargs["dest"] = flag
        aliases = argparse_kwargs.pop("aliases", None)
        name = argparse_kwargs.pop("name", None)  # noqa: F841
        parser.add_argument(*aliases, **argparse_kwargs)

    subparsers = parser.add_subparsers(help="sub-command help", dest=SCAN_KEY)
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
        setup_logger(env.log_file_path())

    vargin = vars(args)

    if vargin[DEBUG_KEY]:
        env.debug(True)

    gpu = vargin[GPU_KEY]

    _logger.debug(vargin)

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # parse and update preferences
    for flag in preferences_flags.keys():
        preferences.set(flag, vargin[flag])

    dicom_path = vargin[DICOM_KEY]
    load_path = vargin[LOAD_KEY]

    if not dicom_path and not load_path:
        raise ValueError("Must provide path to dicoms or path to load data from")

    save_path = vargin[SAVE_KEY]
    if not save_path:
        save_path = load_path if load_path else "%s/data" % dicom_path
        vargin[SAVE_KEY] = save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tissues = parse_tissues(vargin)
    vargin["tissues"] = tissues

    vargin[SPLIT_BY_KEY] = parse_dicom_tag_splitby(vargin[SPLIT_BY_KEY])

    args.func(vargin)

    time_elapsed = time.time() - start_time
    _logger.info("Time Elapsed: {:.2f} seconds".format(time.time() - start_time))

    return time_elapsed


if __name__ == "__main__":
    parse_args()
