"""
Command line interface declaration for knee-related analyses

@author: Arjun Desai
        (C) Stanford University, 2019
"""

import uuid

from tissues.femoral_cartilage import FemoralCartilage
from tissues.meniscus import Meniscus
from tissues.tibial_cartilage import TibialCartilage
from utils.quant_vals import QuantitativeValues as QV

KNEE_KEY = 'knee'
MEDIAL_TO_LATERAL_KEY = 'ml'
TISSUES_KEY = 'tissues'
LOAD_KEY = 'load'
SAVE_KEY = 'save'
PID_KEY = 'pid'
DATA_FORMAT_KEY = 'format'

SUPPORTED_TISSUES = [FemoralCartilage, Meniscus, TibialCartilage]
SUPPORTED_QUANTITATIVE_VALUES = [QV.T2, QV.T1_RHO, QV.T2_STAR]


def knee_parser(base_parser):
    """Parse command line input related to knee

    :param base_parser: the base parser to add knee subcommand to
    """
    parser_tissue = base_parser.add_parser(KNEE_KEY,
                                           help='calculate/analyze quantitative data for knee')

    parser_tissue.add_argument('--%s' % MEDIAL_TO_LATERAL_KEY, action='store_const', const=True, default=False,
                               help="defines slices in sagittal direction going from medial -> lateral")

    parser_tissue.add_argument('--%s' % PID_KEY, nargs='?', default=str(uuid.uuid4()), help='specify pid')

    for tissue in SUPPORTED_TISSUES:
        parser_tissue.add_argument('--%s' % tissue.STR_ID, action='store_const', default=False, const=True,
                                   help='analyze %s' % tissue.FULL_NAME)

    qvs_dict = dict()
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        qv_name = qv.name.lower()
        qvs_dict[qv_name] = qv
        parser_tissue.add_argument('--%s' % qv_name, action='store_const', const=True, default=False,
                                   help='quantify %s' % qv_name)

    parser_tissue.set_defaults(func=handle_knee)


def handle_knee(vargin):
    """Handle parsing command-line input for knee subcommand
    :param vargin:
    :return:
    """
    tissues = vargin[TISSUES_KEY]
    load_path = vargin[LOAD_KEY]
    medial_to_lateral = vargin[MEDIAL_TO_LATERAL_KEY]
    pid = vargin[PID_KEY]

    if tissues is None or len(tissues) == 0:
        print('Computing for all supported knee tissues...')
        tissues = []
        for t in SUPPORTED_TISSUES:
            tissues.append(t())

    # Get all supported quantitative values
    qvs = []
    for qv in SUPPORTED_QUANTITATIVE_VALUES:
        if vargin[qv.name.lower()]:
            qvs.append(qv)

    if len(qvs) == 0:
        print('Computing for all supported quantitative values...')
        qvs = SUPPORTED_QUANTITATIVE_VALUES

    for tissue in tissues:
        tissue.pid = pid
        tissue.medial_to_lateral = medial_to_lateral
        tissue.load_data(load_path)

        print('')
        print('==' * 40)
        print(tissue.FULL_NAME)
        print('==' * 40)

        for qv in qvs:
            # load file
            print('Analyzing %s' % qv.name.lower())
            tissue.calc_quant_vals()

    for tissue in tissues:
        tissue.save_data(vargin[SAVE_KEY], data_format=vargin[DATA_FORMAT_KEY])

    return tissues
