import unittest

import numpy as np

from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import to_affine
from dosma.models.stanford_qdess import StanfordQDessUNet2D

from .. import util as ututils


class TestStanfordQDessUNet2D(unittest.TestCase):
    def test_input_shape(self):
        """Test support for both 3D and 4D inputs."""
        vol = np.ones((256, 256, 2))
        mv = MedicalVolume(
            vol,
            to_affine(("SI", "AP", "LR")),
            headers=ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2}),
        )
        model = StanfordQDessUNet2D(mv.shape[:2] + (1,), weights_path=None)
        out = model.generate_mask(mv)
        assert all(x in out for x in ["pc", "fc", "tc", "men"])
        assert out["pc"].headers().shape == (1, 1, 2)
        del model

        vol = np.stack([np.ones((256, 256, 2)), 2 * np.ones((256, 256, 2))], axis=-1)
        headers = np.stack(
            [
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2}),
                ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 10}),
            ],
            axis=-1,
        )
        mv = MedicalVolume(vol, to_affine(("SI", "AP", "LR")), headers=headers)
        model = StanfordQDessUNet2D(mv.shape[:2] + (1,), weights_path=None)
        out = model.generate_mask(mv)
        assert all(out[x].ndim == 3 for x in ["pc", "fc", "tc", "men"])
        assert out["pc"].headers().shape == (1, 1, 2)

    def test_call(self):
        vol = np.ones((256, 256, 2))
        mv = MedicalVolume(
            vol,
            to_affine(("SI", "AP", "LR")),
            headers=ututils.build_dummy_headers(vol.shape[2], {"EchoTime": 2}),
        )
        model = StanfordQDessUNet2D(mv.shape[:2] + (1,), weights_path=None)
        out = model(mv)
        out2 = model.generate_mask(mv)
        for k in out:
            assert np.all(out[k].volume == out2[k].volume)
        del model
