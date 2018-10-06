from tissues.tissue import Tissue


class PatellarCartilage(Tissue):
    ID = 3
    STR_ID = 'pc'

    def split_regions(self, mask):
        # TODO: implement spliting regions
        pass

    def calc_quant_vals(self, quant_map, mask=None):
        # TODO: implement getting quantitative values for regions regions
        pass
