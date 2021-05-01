import os
import unittest

import h5py
import numpy as np

from dosma.core.io.nifti_io import NiftiReader
from dosma.models.oaiunet2d import IWOAIOAIUnet2D, IWOAIOAIUnet2DNormalized
from dosma.models.seg_model import whiten_volume
from dosma.tissues.femoral_cartilage import FemoralCartilage

import keras.backend as K

from .. import util


@unittest.skipIf(not util.is_data_available(), "unittest data is not available")
class TestIWOAIOAIUnet2D(unittest.TestCase):
    def test_segmentation(self):
        """Check that segmentation works as expected from ported version."""
        classes = ["fc", "tc", "pc", "men"]
        expected_seg = np.load(
            os.path.join(
                util.UNITTEST_DATA_PATH, "datasets/oai/expected/test_001_V00-iwoai-2019-t6.npy"
            )
        )

        scan = NiftiReader().load(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/test_001_V00.nii.gz")
        )

        tissue = FemoralCartilage()
        tissue.find_weights(os.path.join(os.path.dirname(__file__), "../../weights/iwoai-2019-t6"))
        dims = scan.volume.shape
        input_shape = (dims[0], dims[1], 1)
        model = IWOAIOAIUnet2D(input_shape=input_shape, weights_path=tissue.weights_file_path)
        masks = model.generate_mask(scan)
        K.clear_session()

        for i, tissue in enumerate(classes):
            assert np.all(masks[tissue].volume == expected_seg[..., i])


@unittest.skipIf(not util.is_data_available(), "unittest data is not available")
class TestIWOAIOAIUnet2DNormalized(unittest.TestCase):
    def test_h5_nifti_same(self):
        with h5py.File(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00.h5"), "r"
        ) as f:
            h5_data = f["volume"][:]
        scan = NiftiReader().load(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/test_001_V00.nii.gz")
        )
        assert np.all(scan.volume.astype(np.float32) == h5_data)

        # Issue #50 (https://github.com/ad12/DOSMA/issues/50)
        # h5_wh = whiten_volume(np.float32(h5_data))
        # scan_wh = whiten_volume(np.float32(scan.volume))
        # assert np.all(scan_wh == h5_wh)

    def test_normalized_h5df_data(self):
        """Compare zero-mean, unit std. dev. normalization with h5 scans.

        We are doing this comparison to ensure that the normalization done by DOSMA produces the
        expected normalzied output for data stored in h5df files specifically.

        Note that because of precision issues, normalizing arrays of different precision can
        result in different volumes.
        """
        with h5py.File(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00_w.h5"), "r"
        ) as f:
            expected = f["volume"][:]

        with h5py.File(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00.h5"), "r"
        ) as f:
            h5_data = f["volume"][:]
        normalized = whiten_volume(h5_data)

        assert np.all(normalized == expected)

    # def test_same_w(self):
    #     import h5py
    #     from dosma.models.seg_model import whiten_volume
    #     with h5py.File(os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00_w.h5"),
    #                    "r") as f:
    #         whitened = f["volume"][:]
    #
    #     with h5py.File(os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00.h5"), "r") as f:  # noqa: E501
    #         h5_data = f["volume"][:]
    #     h5_data = whiten_volume(h5_data)
    #
    #     assert np.all(whitened == h5_data)
    #
    # def test_same_whitened(self):
    #     import h5py
    #     from dosma.models.seg_model import whiten_volume
    #     with h5py.File(os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/2000001_V00_w.h5"), "r") as f:  # noqa: E501
    #         h5_data = f["volume"][:]
    #
    #     scan = NiftiReader().load(os.path.join(
    #         util.UNITTEST_DATA_PATH, "datasets/oai/test_001_V00.nii.gz"
    #     ))
    #
    #     scan._volume = whiten_volume(scan._volume.astype(np.float32))
    #     assert np.all(scan.volume == h5_data)

    def test_segmentation(self):
        classes = ["fc", "tc", "pc", "men"]
        expected_seg = np.load(
            os.path.join(
                util.UNITTEST_DATA_PATH,
                "datasets/oai/expected/test_001_V00-iwoai-2019-t6-normalized.npy",
            )
        )

        scan = NiftiReader().load(
            os.path.join(util.UNITTEST_DATA_PATH, "datasets/oai/test_001_V00.nii.gz")
        )

        tissue = FemoralCartilage()
        tissue.find_weights(
            os.path.join(os.path.dirname(__file__), "../../weights/iwoai-2019-t6-normalized")
        )
        dims = scan.volume.shape
        input_shape = (dims[0], dims[1], 1)
        model = IWOAIOAIUnet2DNormalized(
            input_shape=input_shape, weights_path=tissue.weights_file_path
        )
        masks = model.generate_mask(scan)
        K.clear_session()

        for i, tissue in enumerate(classes):
            pred = masks[tissue].volume.astype(np.bool)
            gt = expected_seg[..., i].astype(np.bool)
            dice = 2 * np.sum(pred & gt) / np.sum(pred.astype(np.uint8) + gt.astype(np.uint8))
            # Zero-mean normalization of 32-bit vs 64-bit data results in slightly different
            # estimations of the mean and standard deviation.
            # However, when both volumes are compared pre-normalization
            # using np.all() both volumes are the same (see :meth:`test_h5_nifti_same`).
            # This slight difference in the image can affect network performance.
            # As a placeholder, we assert that the dice between the expected and
            # produced segmentations for each tissue must be greater than 99%
            #
            # Update: We found for this particular model and example pair, the segmentations
            # achieve a dice score of 1.0 for all tissues.
            # We enforce that the predicted mask must be equal to the expected mask.
            assert dice >= 0.99, "{}: {:0.6f}".format(tissue, dice)
            assert np.all(
                masks[tissue].volume == expected_seg[..., i]
            ), f"Segmentation not same for {tissue}"
