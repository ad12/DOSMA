from abc import ABC, abstractmethod
import os
from utils import io_utils
from utils.quant_vals import QuantitativeValue
import cv2
import numpy as np

WEIGHTS_FILE_EXT = 'h5'


class Tissue(ABC):
    ID = -1  # should be unique to all tissues, and should not change
    NAME = ''

    def __init__(self, weights_dir = None):
        self.regions = dict()
        self.mask = None
        self.quant_vals = dict()
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
    def calc_quant_vals(self, quant_map, map_type):
        """
        Get quantitative values for tissue
        :param quant_map: a 3D numpy array for quantitative measures (t2, t2*, t1-rho, etc)
        :param map_type: an enum instance of QuantitativeValue
        :return: a dictionary of quantitative values, save in quant_vals
        """
        pass

    def __store_quant_vals__(self, quant_map, quant_df, map_type):
        self.quant_vals[map_type.name] = (quant_map, quant_df)

    def find_weights(self, weights_dir):
        """
        Search for weights file in weights directory
        :param weights_dir: directory where weights are stored
        :return:
        """

        # Find weights file with NAME in the filename, like 'fc_weights.h5'
        files = os.listdir(weights_dir)
        weights_file = None
        for f in files:
            file = os.path.join(weights_dir, f)
            if os.path.isfile(file) and file.endswith(WEIGHTS_FILE_EXT) and self.NAME in file:
                if weights_file is not None:
                    raise ValueError('There are multiple weights files, please remove duplicates')
                weights_file = file

        if weights_file is None:
            raise ValueError('No file found that contains \'%s\' and ends in \'%s\'' % (self.NAME, WEIGHTS_FILE_EXT))

        self.weights_filepath = weights_file

        return weights_file

    def save_data(self, dirpath):
        io_utils.check_dir(dirpath)

        if self.mask:
            mask_filepath = os.path.join(dirpath, '%s.nii.gz' % self.NAME)
            io_utils.save_nifti(mask_filepath, self.mask)

        q_names = []
        dfs = []

        dirpath = os.path.join(dirpath, self.NAME)
        io_utils.check_dir(dirpath)

        for quant_val in QuantitativeValue:
            if quant_val.name not in self.quant_vals:
                continue

            q_names.append(quant_val.name)
            dfs.append(self.quant_vals[1])

            map_filepath = os.path.join(dirpath, quant_val.name + '.tiff')
            q_map = self.quant_vals[0]
            cv2.imwrite(map_filepath, q_map)

        if len(dfs) > 0:
            io_utils.save_tables(os.path.join(dirpath, 'data.xlsx'), dfs, q_names)

    def load_data(self, dirpath):
        # load mask, if no mask exists stop loading information
        mask_filepath = os.path.join(dirpath, '%s.nii.gz' % self.NAME)
        if not os.path.isfile(mask_filepath):
            raise FileNotFoundError('File \'%s\' does not exist' % mask_filepath)

        filepath = os.path.join(dirpath, '%s.nii.gz' % self.NAME)
        self.mask = io_utils.load_nifti(filepath)


    def __data_filename__(self):
        return '%s.%s' % (self.NAME, io_utils.DATA_EXT)