import os
import unittest
import warnings

import numpy as np

from dosma.core.io.nifti_io import NiftiWriter
from dosma.core.med_volume import MedicalVolume
from dosma.scan_sequences.mri import Mapss
from dosma.tissues.femoral_cartilage import FemoralCartilage

from ... import util

SEGMENTATION_WEIGHTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "../../weights/iwoai-2019-t6-normalized"
)
SEGMENTATION_MODEL = "iwoai-2019-t6-normalized"

# Path to manual segmentation mask
MANUAL_SEGMENTATION_MASK_PATH = os.path.join(
    util.get_scan_dirpath(Mapss.NAME), "misc/fc_manual.nii.gz"
)


class MapssTest(util.ScanTest):
    SCAN_TYPE = Mapss

    def _generate_mock_data(self, shape=None, ts=None, metadata=True):
        """Generates mock monexponential data for MAPSS sequence.

        The mock data is overly simplified in that the t1rho and t2 maps
        are identical. This is to simply the data generation process.

        Args:
            ys: The volumes at different spin lock times.
            ts: Echo times.
            t1rho (ndarray): The t1rho times for each voxel
            a (ndarray or int): The multiplicative constant.
        """
        if shape is None:
            shape = (10, 10, 10)
        if ts is None:
            ts = [0, 10, 12.847, 25.695, 40, 51.39, 80]
        else:
            assert len(ts) == 7

        a = 1.0
        t2 = t1rho = np.random.rand(*shape) * 80 + 0.1
        _, ys, _, _ = util.generate_monoexp_data(shape=shape, x=ts, a=a, b=-1 / t1rho)

        if metadata:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for idx, (y, t) in enumerate(zip(ys, ts)):
                    y.set_metadata("EchoTime", t, force=True)
                    y.set_metadata("EchoNumber", idx + 1, force=True)

        return ys, ts, a, t1rho, t2

    def test_generate_t1_rho_map(self):
        ys, _, _, _, _ = self._generate_mock_data()
        scan = Mapss(ys)

        mask = MedicalVolume(np.ones(ys[0].shape), np.eye(4))
        mask_path = os.path.join(self.data_dirpath, "test_t1rho_mask.nii.gz")
        NiftiWriter().save(mask, mask_path)

        tissue = FemoralCartilage()
        map1 = scan.generate_t1_rho_map(tissue)
        assert map1 is not None

        tissue.set_mask(mask)
        map2 = scan.generate_t1_rho_map(tissue, num_workers=util.num_workers())
        assert map2 is not None
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

        map2 = scan.generate_t1_rho_map(tissue, mask_path=mask, num_workers=util.num_workers())
        assert map2 is not None
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

    def test_generate_t2_map(self):
        ys, _, _, _, _ = self._generate_mock_data()
        scan = Mapss(ys)

        mask = MedicalVolume(np.ones(ys[0].shape), np.eye(4))
        mask_path = os.path.join(self.data_dirpath, "test_t2_mask.nii.gz")
        NiftiWriter().save(mask, mask_path)

        tissue = FemoralCartilage()
        map1 = scan.generate_t2_map(tissue)
        assert map1 is not None

        tissue.set_mask(mask)
        map2 = scan.generate_t2_map(tissue, num_workers=util.num_workers())
        assert map2 is not None
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

        map2 = scan.generate_t2_map(tissue, mask_path=mask, num_workers=util.num_workers())
        assert map2 is not None
        assert map1.volumetric_map.is_identical(map2.volumetric_map)

    @unittest.skipIf(
        not util.is_data_available() or not util.is_elastix_available(),
        "unittest data or elastix is not available",
    )
    def test_cmd_line(self):
        # Intraregister
        cmdline_str = "--d %s --s %s mapss intraregister" % (self.dicom_dirpath, self.data_dirpath)
        self.__cmd_line_helper__(cmdline_str)

        # Estimate T1-rho for femoral cartilage.
        cmdline_str = "--l %s mapss --fc t1_rho --mask %s" % (
            self.data_dirpath,
            MANUAL_SEGMENTATION_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)

        # Generate T2 map for femoral cartilage, tibial cartilage, and meniscus via command line
        cmdline_str = "--l %s mapss --fc t2 --mask %s" % (
            self.data_dirpath,
            MANUAL_SEGMENTATION_MASK_PATH,
        )
        self.__cmd_line_helper__(cmdline_str)
