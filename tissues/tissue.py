from abc import ABC, abstractmethod
import os
from utils import im_utils

WEIGHTS_FILE_EXT = 'h5'

class Tissue(ABC):
    ID = -1  # should be unique to all tissues, and should not change
    NAME = ''

    def __init__(self, weights_dir = None):
        self.regions = None
        self.mask = None
        self.quant_vals = None
        self.weights_filepath = None

        if (weights_dir is not None):
            self.weights_filepath = self.find_weights(weights_dir)

    @abstractmethod
    def split_regions(self, mask):
        """
        Split mask into regions
        :param mask:
        :return: a 4D numpy array (region, height, width, depth) - save in variable self.regions
        """
        pass

    @abstractmethod
    def calc_quant_vals(self, map, mask=None):
        """
        Get quantitative values for tissue
        :param map: a 3D numpy array for quantitative measures (t2, t2*, t1-rho, etc)
        :param mask: a 3D binary numpy array if split_regions is not called
        :return: a dictionary of quantitative values, save in quant_vals
        """
        pass

    def find_weights(self, weights_dir):
        """
        Search for weights file in weights directory
        :param weights_dir: directory where weights are stored
        :return:
        """

        # Find weights file with NAME in the filename, like 'fc_weights.h5'
        files = os.listdir(weights_dir)
        weights_file = None
        for file in files:
            if os.path.isfile(file) and file.endswith(WEIGHTS_FILE_EXT) and self.NAME in file:
                if weights_file is not None:
                    raise ValueError('There are multiple weights files, please remove duplicates')
                weights_file = file

        if weights_file is None:
            raise ValueError('No file found that contains \'%s\' and ends in \'%s\'' % (self.NAME, WEIGHTS_FILE_EXT))

        self.weights_filepath = weights_file

    def save_data(self, dirpath):
        dirpath = os.path.join(dirpath, self.NAME)

        # save mask
        mask_filepath = os.path.join(dirpath, '%s.%s' % ('mask', 'tiff'))
        im_utils.write_3d(mask_filepath, self.mask)

        # TODO: save quantitative maps

        # TODO: write quantitative values as table