import numpy as np
from tissues.tissue import Tissue

class FemoralCartilage(Tissue):
    ID = 3
    NAME = 'pc'

    def split_regions(self, mask):
        #TODO: implement spliting regions
        pass

    def calc_quant_vals(self, quant_map, mask=None):
        # TODO: implement getting quantitative values for regions regions
        pass