import os
import unittest
import warnings

import numpy as np
from pydicom.tag import Tag

from dosma.core.med_volume import MedicalVolume
from dosma.core.quant_vals import QuantitativeValue
from dosma.models.model_loading_util import get_model
from dosma.scan_sequences.mri.qdess import QDess
from dosma.tissues.femoral_cartilage import FemoralCartilage

import tensorflow.keras.backend as K

from ... import util

SEGMENTATION_WEIGHTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "../../../weights/iwoai-2019-t6-normalized"
)
SEGMENTATION_MODEL = "iwoai-2019-t6-normalized"


class QDessTest(util.ScanTest):
    SCAN_TYPE = QDess

    def _generate_mock_data(self, shape=None, metadata=True):
        """Generates arbitrary mock data for QDess sequence.

        Metadata values were extracted from a real qDESS sequence.
        """
        if shape is None:
            shape = (10, 10, 10)
        e1 = MedicalVolume(np.random.rand(*shape) * 80 + 0.1, affine=np.eye(4))
        e2 = MedicalVolume(np.random.rand(*shape) * 40 + 0.1, affine=np.eye(4))
        ys = [e1, e2]
        ts = [8, 42]
        if metadata:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for idx, (y, t) in enumerate(zip(ys, ts)):
                    y.set_metadata("EchoTime", t, force=True)
                    y.set_metadata("EchoNumber", idx + 1, force=True)
                    y.set_metadata("RepetitionTime", 25.0, force=True)
                    y.set_metadata("FlipAngle", 30.0, force=True)
                    y.set_metadata(Tag(0x001910B6), 3132.0, force=True)  # gradient time
                    y.set_metadata(Tag(0x001910B7), 1560.0, force=True)  # gradient area

        return ys, ts, None

    def test_basic(self):
        ys, _, _ = self._generate_mock_data()
        scan = QDess(ys)
        assert scan.ref_dicom == ys[0].headers(flatten=True)[0]

        with self.assertRaises(ValueError):
            _ = QDess(ys + ys)

    def test_calc_rss(self):
        ys, _, _ = self._generate_mock_data()
        scan = QDess(ys)
        rss = scan.calc_rss()

        assert np.allclose(rss.A, np.sqrt(ys[0] ** 2 + ys[1] ** 2).A)

    def test_generate_t2_map(self):
        ys, _, _ = self._generate_mock_data()
        scan = QDess(ys)

        tissue = FemoralCartilage()
        t2 = scan.generate_t2_map(tissue)
        assert isinstance(t2, QuantitativeValue)

    def test_save_load(self):
        ys, _, _ = self._generate_mock_data()
        scan = QDess(ys)

        save_dir = os.path.join(self.data_dirpath, "test_save_load")
        pik_file = scan.save(save_dir, save_custom=True)
        assert os.path.isfile(pik_file)
        assert all(
            os.path.isfile(os.path.join(save_dir, "volumes", f"echo-{idx:03d}.nii.gz"))
            for idx in range(2)
        )

        scan2 = QDess.load(pik_file)
        for v1, v2 in zip(scan.volumes, scan2.volumes):
            assert v1.is_identical(v2)

    @unittest.skipIf(not util.is_data_available(), "unittest data is not available")
    def test_segmentation_multiclass(self):
        """Test support for multiclass segmentation."""
        scan = self.SCAN_TYPE.from_dicom(self.dicom_dirpath, num_workers=util.num_workers())
        tissue = FemoralCartilage()
        tissue.find_weights(SEGMENTATION_WEIGHTS_FOLDER),  # noqa B018
        dims = scan.get_dimensions()
        input_shape = (dims[0], dims[1], 1)
        model = get_model(
            SEGMENTATION_MODEL, input_shape=input_shape, weights_path=tissue.weights_file_path
        )
        scan.segment(model, tissue, use_rss=True)

        # This should call __del__ in KerasSegModel
        model = None
        K.clear_session()

    @unittest.skipIf(not util.is_data_available(), "unittest data is not available")
    def test_cmd_line(self):
        # Generate segmentation mask for femoral cartilage via command line.
        cmdline_str = (
            f"--d {self.dicom_dirpath} --s {self.data_dirpath} qdess --fc "
            f"segment --weights_dir {SEGMENTATION_WEIGHTS_FOLDER} "
            f"--model {SEGMENTATION_MODEL} --use_rss"
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2 map for femoral cartilage, tibial cartilage, and meniscus.
        cmdline_str = (
            f"--l {self.data_dirpath} qdess --fc t2 --suppress_fat " f"--suppress_fluid --beta 1.1"
        )
        self.__cmd_line_helper__(cmdline_str)


if __name__ == "__main__":
    unittest.main()
