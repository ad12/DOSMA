from tissues.tissue import Tissue
import scipy.ndimage as sni
import numpy as np
from med_objects.med_volume import MedicalVolume
import pandas as pd
import matplotlib.pyplot as plt
from utils.quant_vals import QuantitativeValues
from utils import io_utils
import warnings
import defaults
import os

# milliseconds
BOUNDS = {QuantitativeValues.T2: 60.0,
          QuantitativeValues.T1_RHO: 100.0,
          QuantitativeValues.T2_STAR: 50.0}

__all__ = ['PatellarCartilage']

class PatellarCartilage(Tissue):
    """Handles analysis for patellar cartilage"""
    ID = 2
    STR_ID = 'pc'
    FULL_NAME = 'patellar cartilage'

    # Expected quantitative values
    T1_EXPECTED = 1000  # milliseconds

    # Coronal Keys
    ANTERIOR_KEY = 0
    POSTERIOR_KEY = 1
    CORONAL_KEYS = [ANTERIOR_KEY, POSTERIOR_KEY]

    # Saggital Keys
    MEDIAL_KEY = 0
    LATERAL_KEY = 1
    SAGGITAL_KEYS = [MEDIAL_KEY, LATERAL_KEY]

    # Axial Keys
    SUPERIOR_KEY = 0
    INFERIOR_KEY = 1
    TOTAL_AXIAL_KEY = -1

    def __init__(self, weights_dir=None, medial_to_lateral=None):
        """
        :param weights_dir: Directory to weights files
        :param medial_to_lateral: True or False, if false, then lateral to medial
        """
        super().__init__(weights_dir=weights_dir)

        self.regions_mask = None
        self.medial_to_lateral = medial_to_lateral

    def unroll_coronal(self, quant_map):
        mask = self.__mask__.volume

        assert self.regions_mask is not None, "region_mask not initialized. Should be initialized when mask is set"
        region_mask_sup_inf = self.regions_mask[..., 0]

        superior = (region_mask_sup_inf == self.SUPERIOR_KEY) * mask * quant_map
        superior[superior == 0] = np.nan
        superior = np.nanmean(superior, axis=0)

        inferior = (region_mask_sup_inf == self.INFERIOR_KEY) * mask * quant_map
        inferior[inferior == 0] = np.nan
        inferior = np.nanmean(inferior, axis=0)

        total = mask * quant_map
        total[total == 0] = np.nan
        total = np.nanmean(total, axis=0)

        return total, superior, inferior

    def split_regions(self, base_map):
        center_of_mass = sni.measurements.center_of_mass(base_map) # zero indexed

        if np.sum(base_map) == 0:
            warnings.warn('No mask for `%s` was found.' % self.FULL_NAME)

    def __calc_quant_vals__(self, quant_map, map_type):
        pass

    def set_mask(self, mask):
        mask_copy = MedicalVolume(mask.volume, mask.pixel_spacing)

        super().set_mask(mask_copy)

        self.split_regions(self.__mask__.volume)

    def __save_quant_data__(self, dirpath):
        """Save quantitative data and 2D visualizations of patellar cartilage

               Check which quantitative values (T2, T1rho, etc) are defined for patellar cartilage and analyze these
               1. Save 2D total, superficial, and deep visualization maps
               2. Save {'medial', 'lateral'}, {'anterior', 'posterior'}, {'superior', 'inferior', 'total'} data to excel
                       file

               :param dirpath: base filepath to save data
               """
        pass
