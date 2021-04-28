import unittest

import numpy as np

from dosma.core.io.format_io import ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter
from dosma.core.med_volume import MedicalVolume
from dosma.core.orientation import to_affine

from .. import util as ututils


class TestOrientation(unittest.TestCase):
    nr = NiftiReader()
    nw = NiftiWriter()

    def setUp(self):
        # filepaths to all separated echos
        # (done in image-viewer such as ITK-Snap, Horos, etc)
        filepaths = []
        for fp in ututils.SCAN_DIRPATHS:
            filepaths.extend(ututils.get_read_paths(fp, ImageDataFormat.nifti))
        self.filepaths = filepaths

    def check_orientations(self, mv: MedicalVolume, orientations):
        """
        Apply each orientation specified in orientations to the Medical Volume mv
        Assert if mv --> apply orientation --> apply original orientation != mv original
        position coordinates.

        Args:
            mv: a Medical Volume
            orientations: a list or tuple of orientation tuples
        """
        o_base, so_base, ps_base = mv.orientation, mv.scanner_origin, mv.pixel_spacing
        ps_affine = np.array(mv.affine)

        for o in orientations:
            # Reorient to some orientation
            mv.reformat(o, inplace=True)

            # Reorient to original orientation
            mv.reformat(o_base, inplace=True)

            assert mv.orientation == o_base, "Orientation mismatch: Expected %s, got %s" % (
                str(o_base),
                str(mv.orientation),
            )
            assert mv.scanner_origin == so_base, "Scanner Origin mismatch: Expected %s, got %s" % (
                str(so_base),
                str(mv.scanner_origin),
            )
            assert mv.pixel_spacing == ps_base, "Pixel Spacing mismatch: Expected %s, got %s" % (
                str(ps_base),
                str(mv.pixel_spacing),
            )

            assert (
                mv.affine == ps_affine
            ).all(), "Affine matrix mismatch: Expected\n%s\ngot\n%s" % (
                str(ps_affine),
                str(mv.affine),
            )

    def test_flip(self):
        # tests flipping orientation across volume axis
        for fp in self.filepaths:
            e1 = self.nr.load(fp)
            o = e1.orientation
            orientations = [
                (o[0][::-1], o[1], o[2]),
                (o[0], o[1][::-1], o[2]),
                (o[0], o[1][::-1], o[2]),
                (o[0][::-1], o[1][::-1], o[2]),
                (o[0][::-1], o[1], o[2][::-1]),
                (o[0], o[1][::-1], o[2][::-1]),
                (o[0][::-1], o[1][::-1], o[2][::-1]),
            ]

            self.check_orientations(e1, orientations)

    def test_transpose(self):
        # tests transposing axes - i.e. changing vantage point on current volume
        for fp in self.filepaths:
            e1 = self.nr.load(fp)
            o = e1.orientation

            orientations = [
                (o[1], o[2], o[0]),
                (o[2], o[0], o[1]),
                (o[1], o[0], o[2]),
                (o[2], o[1], o[0]),
                (o[0], o[2], o[1]),
            ]

            self.check_orientations(e1, orientations)

    def test_transpose_and_flip(self):
        from itertools import permutations

        for fp in self.filepaths:
            e1 = self.nr.load(fp)
            o = e1.orientation

            # generate all possible transposed versions of the existing orientation
            transpose_orientations = list(permutations(o))
            flip_orientations_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
            orientations = []
            for to in transpose_orientations:
                for fo_inds in flip_orientations_indices:
                    o_test = list(to)
                    for i in fo_inds:
                        o_test[i] = o_test[i][::-1]
                    orientations.append(tuple(o_test))

            self.check_orientations(e1, orientations)


class TestToAffine(unittest.TestCase):
    """Test cases for dosma.io.orientation.to_affine."""

    def test_basic(self):
        """Basic transposes and flips in RAS+ coordinate system."""
        orientations = [
            ("LR", "PA", "IS"),  # standard RAS+
            ("RL", "AP", "SI"),  # flipped
            ("IS", "LR", "PA"),  # transposed
            ("AP", "SI", "RL"),  # transposed + flipped
        ]

        for ornt in orientations:
            affine = to_affine(ornt)
            mv = MedicalVolume(np.ones((10, 20, 30)), affine)
            assert mv.orientation == ornt
            assert np.all(np.asarray(mv.pixel_spacing) == 1)
            assert np.all(np.asarray(mv.scanner_origin) == 0)

    def test_spacing(self):
        """Test affine matrix with pixel spacing."""
        ornt = ("AP", "SI", "RL")

        spacing = np.random.rand(3) + 0.1  # avoid pixel spacing of 0
        affine = to_affine(ornt, spacing)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine)
        assert mv.orientation == ornt
        assert np.all(np.asarray(mv.pixel_spacing) == spacing)
        assert np.all(np.asarray(mv.scanner_origin) == 0)

        spacing = np.random.rand(1) + 0.1  # avoid pixel spacing of 0
        expected_spacing = np.asarray(list(spacing) + [1.0, 1.0])
        affine = to_affine(ornt, spacing)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine)
        assert mv.orientation == ornt
        assert np.all(np.asarray(mv.pixel_spacing) == expected_spacing)
        assert np.all(np.asarray(mv.scanner_origin) == 0)

    def test_origin(self):
        """Test affine matrix with scanner origin."""
        ornt = ("AP", "SI", "RL")

        origin = np.random.rand(3)
        affine = to_affine(ornt, spacing=None, origin=origin)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine)
        assert mv.orientation == ornt
        assert np.all(np.asarray(mv.pixel_spacing) == 1)
        assert np.all(np.asarray(mv.scanner_origin) == origin)

        origin = np.random.rand(1)
        expected_origin = np.asarray(list(origin) + [0.0, 0.0])
        affine = to_affine(ornt, spacing=None, origin=origin)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine)
        assert mv.orientation == ornt
        assert np.all(np.asarray(mv.pixel_spacing) == 1)
        assert np.all(np.asarray(mv.scanner_origin) == expected_origin)

    def test_complex(self):
        ornt = ("AP", "SI", "RL")
        spacing = (0.5, 0.7)
        origin = (100, -54)

        expected_spacing = np.asarray(list(spacing) + [1.0])
        expected_origin = np.asarray(list(origin) + [0.0])

        affine = to_affine(ornt, spacing=spacing, origin=origin)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine)
        assert mv.orientation == ornt
        assert np.all(np.asarray(mv.pixel_spacing) == expected_spacing)
        assert np.all(np.asarray(mv.scanner_origin) == expected_origin)


if __name__ == "__main__":
    unittest.main()
