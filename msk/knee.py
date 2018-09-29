from tissues.femoral_cartilage import FemoralCartilage
from utils.quant_vals import QuantitativeValue as QV
from utils import io_utils

KNEE_KEY = 'knee'
ORIENTATION_KEY = 'orientation'
TISSUES_KEY = 'tissues'
LOAD_KEY = 'load'
SAVE_KEY = 'save'

SUPPORTED_TISSUES = [FemoralCartilage()]
SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]


def knee_parser(subparsers):
    parser_tissue = subparsers.add_parser(KNEE_KEY, help='calculate/analyze quantitative data for MSK knee')
    parser_tissue.add_argument('--%s' % ORIENTATION_KEY, choices={'LEFT', 'RIGHT'}, nargs='?', default='RIGHT',
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
    tissues = vargin[TISSUES_KEY]
    load_path = vargin[LOAD_KEY]
    orientation = vargin[ORIENTATION_KEY]

    # Get all supported quantitative values
    qvs = []
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        if vargin[qv.name.lower()]:
            qvs.append(qv)

    for tissue in tissues:
        tissue.ORIENTATION = orientation
        tissue.load_data(load_path)

        for qv in qvs:
            # load file
            filepath = find_filepath_with_qv(load_path, qv)
            tmp = io_utils.load_h5(filepath)
            qv_map = tmp['data']
            print(qv.name)
            tissue.calc_quant_vals(qv_map, qv)

    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY])

    return tissues


def find_filepath_with_qv(load_path, qv):
    import glob, os
    dirlist = glob.glob(os.path.join(load_path, '*', '%s.h5' % qv.name.lower()))

    name = qv.name.lower()

    if len(dirlist) == 0:
        raise ValueError('No map for %s found. Must have name %s.h5' % (name, name))

    if (len(dirlist) > 1):
        raise ValueError('Multiple %s maps found. Delete extra %s maps' % (name, name))

    return dirlist[0]
