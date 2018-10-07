from tissues.femoral_cartilage import FemoralCartilage
from utils import io_utils
from utils.quant_vals import QuantitativeValues as QV

KNEE_KEY = 'knee'
DIRECTION_KEY = 'direction'
TISSUES_KEY = 'tissues'
LOAD_KEY = 'load'
SAVE_KEY = 'save'

SUPPORTED_TISSUES = [FemoralCartilage()]
SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]


def knee_parser(base_parser):
    """Parse command line input related to knee

    :param base_parser: the base parser to add knee subcommand to
    """
    parser_tissue = base_parser.add_parser(KNEE_KEY, help='calculate/analyze quantitative data for MSK knee')
    parser_tissue.add_argument('--%s' % DIRECTION_KEY, choices={'LEFT', 'RIGHT'}, nargs='?', default='RIGHT',
                               help='knee orientation (left or right)')

    for tissue in SUPPORTED_TISSUES:
        parser_tissue.add_argument('-%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                                   help='analyze %s' % tissue.FULL_NAME)

    qvs_dict = dict()
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        qv_name = qv.name.lower()
        qvs_dict[qv_name] = qv
        parser_tissue.add_argument('-%s' % qv_name, action='store_const', const=True, default=False,
                                   help='quantify %s' % qv_name)

        parser_tissue.set_defaults(func=handle_knee)


def handle_knee(vargin):
    """Handle parsing command-line input for knee subcommand
    :param vargin:
    :return:
    """
    tissues = vargin[TISSUES_KEY]
    load_path = vargin[LOAD_KEY]
    direction = vargin[DIRECTION_KEY]

    # Get all supported quantitative values
    qvs = []
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        if vargin[qv.name.lower()]:
            qvs.append(qv)

    for tissue in tissues:
        tissue.knee_direction = direction
        tissue.load_data(load_path)

        print('')
        print('==' * 40)
        print(tissue.FULL_NAME)
        print('==' * 40)

        for qv in qvs:
            # load file
            print('Analyzing %s' % qv.name.lower())
            filepath = find_filepath_with_qv(load_path, qv)
            tmp = io_utils.load_nifti(filepath)
            tissue.calc_quant_vals(tmp, qv)

    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY])

    return tissues


def find_filepath_with_qv(load_path, qv):
    """Find filepath to the quantitative value map.

    All maps must be stored in the nifti format

    :param load_path: base path for searching
    :param qv: a QuantiativeValue
    :return: a path to the quantitative value map

    :raise ValueError:
                1. No files (recursively searched) in load_path directory
                2. Multiple files found for the same quantitative value
    """
    import glob, os
    dirlist = glob.glob(os.path.join(load_path, '*', '%s.nii.gz' % qv.name.lower()))

    name = qv.name.lower()

    if len(dirlist) == 0:
        raise ValueError('No map for %s found. Must have name %s.nii.gz' % (name, name))

    if len(dirlist) > 1:
        raise ValueError('Multiple %s maps found. Delete extra %s maps' % (name, name))

    return dirlist[0]
