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
    parser_tissue = subparsers.add_parser(KNEE_KEY, help='analyze tissues')
    parser_tissue.add_argument('--%s' % ORIENTATION_KEY, choices={'LEFT', 'RIGHT'}, nargs='?', default='RIGHT')

    for tissue in SUPPORTED_TISSUES:
        parser_tissue.add_argument('-%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                                   help='handle %s' % tissue.FULL_NAME)

    qvs_dict = dict()
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        qv_name = qv.name.lower()
        qvs_dict[qv_name] = qv
        parser_tissue.add_argument('-%s' % qv_name, nargs=1, default=None,
                                   help='calculate %s' % qv_name)

        parser_tissue.set_defaults(func=handle_knee)


def handle_knee(vargin):
    tissues = vargin[TISSUES_KEY]
    load_path = vargin[LOAD_KEY]
    orientation = vargin[ORIENTATION_KEY]

    # Get all supported quantitative values
    qvs = []
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        if vargin[qv.name.lower()]:
            filepath = vargin[qv.name.lower()][0]
            qvs.append((qv, filepath))

    for tissue in tissues:
        tissue.ORIENTATION = orientation
        tissue.load_data(load_path)

        for qv, filepath in qvs:
            # load file
            tmp = io_utils.load_h5(filepath)
            qv_map = tmp['data']

            tissue.calc_quant_vals(qv_map, qv)

    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY])

    return tissues