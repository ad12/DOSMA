import numpy as np
from tissues.tissue import Tissue

class FemoralCartilage(Tissue):
    ID = 4
    NAME = 'men'

    # Region keys
    DEEP_MEDIAL_REGION_KEY = 'deep_medial'
    DEEP_CENTRAL_REGION_KEY = 'deep_central'
    DEEP_LATERAL_REGION_KEY = 'deep_lateral'
    SUPERFICIAL_MEDIAL_REGION_KEY = 'superficial_medial'
    SUPERFICIAL_CENTRAL_REGION_KEY = 'superficial_central'
    SUPERFICIAL_LATERAL_REGION_KEY = 'superficial_lateral'


    def split_regions(self, mask):
        #TODO: implement spliting regions
        pass

    def calc_quant_vals(self, quant_map, mask=None):
        # TODO: implement getting quantitative values for regions regions
        pass