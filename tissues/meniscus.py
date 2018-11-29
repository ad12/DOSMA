from tissues.tissue import Tissue


class Meniscus(Tissue):
    """Handles analysis for meniscus"""
    ID = 2
    STR_ID = 'men'
    FULL_NAME = 'meniscus'

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        """
        :param weights_dir: Directory to weights files
        :param medial_to_lateral: True or False, if false, then lateral to medial
        """
        super().__init__(weights_dir=weights_dir)

        self.regions_mask = None
        self.medial_to_lateral = medial_to_lateral

