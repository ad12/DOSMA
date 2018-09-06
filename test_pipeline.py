import unittest
import models
import keras.backend as K
import generate_mask
import numpy as np
import scipy.io as sio

import utils
import dess_utils


DICOMS_PATH = '../dicoms/10263603'
DESS_003_DICOM_PATH = '../dicoms/003'
DESS_003_T2_MAP_PATH = '../dicoms/003_t2_map.mat'
DECIMAL_PRECISION = 3 # (+/- 0.001)


class ModelsTest(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_load_weights(self):
        """Test loading weights into keras1/keras2 models"""
        input_size = (288, 288, 1)
        weights_paths = list(models.WEIGHTS_DICT.values())

        model = models.unet_2d_model(input_size)

        for weights_path in weights_paths:
            model.load_weights(weights_path, by_name=True)
        K.clear_session()

        model = models.unet_2d_model_keras2(input_size)
        for weights_path in weights_paths:
            model.load_weights(weights_path, by_name=True)

        K.clear_session()

    def test_keras_version_models(self):
        """Test keras/keras2 models produce the same output"""
        volume, _ = utils.load_dicom(DICOMS_PATH)
        volume = utils.whiten_volume(volume)
        im_shape = volume.shape
        # reshape volume to be (slice, x, y, 1)
        v = np.transpose(volume, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)

        # original model output
        model = models.unet_2d_model((im_shape[0], im_shape[1], 1))
        tissue_weight_path = models.WEIGHTS_DICT['fc']
        model.load_weights(tissue_weight_path, by_name=True)
        mask_orig = model.predict(v, batch_size=32)
        K.clear_session()

        # keras 2.0 model output
        model = models.unet_2d_model_keras2((im_shape[0], im_shape[1], 1))
        tissue_weight_path = models.WEIGHTS_DICT['fc']
        model.load_weights(tissue_weight_path, by_name=True)
        mask_new = model.predict(v, batch_size=32)
        K.clear_session()

        assert((mask_orig == mask_new).all())

    def test_batch_size(self):
        """Test if batch size makes a difference on the output"""
        volume, _ = utils.load_dicom(DICOMS_PATH)
        volume = utils.whiten_volume(volume)
        im_shape = volume.shape
        # reshape volume to be (slice, x, y, 1)
        v = np.transpose(volume, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)

        # keras 2.0 model output
        model = models.unet_2d_model_keras2((im_shape[0], im_shape[1], 1))
        tissue_weight_path = models.WEIGHTS_DICT['fc']
        model.load_weights(tissue_weight_path, by_name=True)

        mask_1 = model.predict(v, batch_size=1)
        mask_4 = model.predict(v, batch_size=4)
        mask_16 = model.predict(v, batch_size=16)

        K.clear_session()

        assert (mask_1 == mask_4).all()
        assert (mask_1 == mask_16).all()

    def test_generate_mask_inputs(self):
        """Test incorrect inputs in models.generate_mask(volume, tissue)"""

        # Test volume
        volume1 = np.zeros([80, 288, 30]) # height >= 288
        volume2 = np.zeros([288, 80, 30]) # width >= 288
        volume3 = [] # not valid numpy array
        volume4 = 10 # not valid numpy array
        volume5 = np.zeros([300, 400]) # not valid dimensions
        tissue = 'fc'

        volumes = [volume1, volume2, volume3, volume4, volume5]

        for v in volumes:
            with self.assertRaises(ValueError):
                models.generate_mask(v, tissue)

        # Test tissue
        tissue1 = 'a' # not valid string
        tissue2 = 'helo' # not valid string
        tissue3 = 4 # must be string
        tissue4 = [models.FEMORAL_CARTILAGE_STR, models.TIBIAL_CARTILAGE_STR] # should not handle lists
        volume = np.zeros([500, 500, 3])

        tissues = [tissue1, tissue2, tissue3, tissue4]

        with self.assertRaises(ValueError):
            models.generate_mask(volume, tissue1)

        with self.assertRaises(ValueError):
            models.generate_mask(volume, tissue2)

        with self.assertRaises(ValueError):
            models.generate_mask(volume, tissue3)

        with self.assertRaises(ValueError):
            models.generate_mask(volume, tissue4)

    def test_unet_2d_generation_inputs(self):
        """Test inputs into 2d unet generation """
        input_size1 = 10
        input_size2 = [300, 400, 3]
        input_size3 = (400, 500)
        input_size4 = (400, 500, 3)

        with self.assertRaises(ValueError):
            models.unet_2d_model_keras2(input_size1)

        with self.assertRaises(ValueError):
            models.unet_2d_model_keras2(input_size2)

        with self.assertRaises(ValueError):
            models.unet_2d_model_keras2(input_size3)

        with self.assertRaises(ValueError):
            models.unet_2d_model_keras2(input_size4)


class GenerateMaskTest(unittest.TestCase):
    pass


class T2_Map_Test(unittest.TestCase):
    def test_t2_map(self):
        # Load ground truth t2 map
        mat_t2_map = sio.loadmat(DESS_003_T2_MAP_PATH)
        mat_t2_map = mat_t2_map['t2map']

        # calculate t2 map
        dicom_ext = 'dcm'
        dicom_array, ref_dicom = utils.load_dicom(DESS_003_DICOM_PATH, dicom_ext)
        py_t2_map = dess_utils.calc_t2_map(dicom_array, ref_dicom)

        # need to convert all np.nan values to 0 before comparing
        # np.nan does not equal np.nan, so we need the same values to compare
        mat_t2_map = np.nan_to_num(mat_t2_map)
        py_t2_map = np.nan_to_num(py_t2_map)

        # Round to the nearest 1000th (0.001)
        mat_t2_map = np.round(mat_t2_map, decimals=DECIMAL_PRECISION)
        py_t2_map = np.round(py_t2_map, decimals=DECIMAL_PRECISION)

        assert((mat_t2_map == py_t2_map).all())



if __name__ == '__main__':
    unittest.main()
