import unittest
import numpy as np
import sys

sys.path.append('../')
from unit_tests import unittest_utils as ututils
from data_io.format_io import ImageDataFormat


class TestOrientation(unittest.TestCase):
    from data_io.nifti_io import NiftiReader, NiftiWriter
    from data_io.med_volume import MedicalVolume
    nr = NiftiReader()
    nw = NiftiWriter()

    def setUp(self):
        # filepaths to all separated echos (done in image-viewer such as ITK-Snap, Horos, etc)
        filepaths = []
        for fp in ututils.SCAN_DIRPATHS:
            filepaths.extend(ututils.get_read_paths(fp, ImageDataFormat.nifti))
        self.filepaths = filepaths

    def check_orientations(self, mv: MedicalVolume, orientations):
        """
        Apply each orientation specified in orientations to the Medical Volume mv
        Assert if mv --> apply orientation --> apply original orientation != mv original position coordinates
        :param mv: a Medical Volume
        :param orientations: a list or tuple of orientation tuples
        """
        o_base, so_base, ps_base = mv.orientation, mv.scanner_origin, mv.pixel_spacing
        ps_affine = np.array(mv.affine)

        for o in orientations:
            # Reorient to some orientation
            mv.reformat(o)

            # Reorient to original orientation
            mv.reformat(o_base)

            assert mv.orientation == o_base, "Orientation mismatch: Expected %s, got %s" % (str(o_base),
                                                                                            str(mv.orientation))
            assert mv.scanner_origin == so_base, "Scanner Origin mismatch: Expected %s, got %s" % (str(so_base),
                                                                                                   str(
                                                                                                       mv.scanner_origin))
            assert mv.pixel_spacing == ps_base, "Pixel Spacing mismatch: Expected %s, got %s" % (str(ps_base),
                                                                                                 str(mv.pixel_spacing))

            assert (mv.affine == ps_affine).all(), "Affine matrix mismatch: Expected\n%s\ngot\n%s" % (str(ps_affine),
                                                                                                      str(mv.affine))

    def test_flip(self):
        # tests flipping orientation across volume axis
        for fp in self.filepaths:
            e1 = self.nr.load(fp)
            o = e1.orientation
            orientations = [(o[0][::-1], o[1], o[2]), (o[0], o[1][::-1], o[2]), (o[0], o[1][::-1], o[2]),
                            (o[0][::-1], o[1][::-1], o[2]), (o[0][::-1], o[1], o[2][::-1]), (o[0], o[1][::-1], o[2][::-1]),
                            (o[0][::-1], o[1][::-1], o[2][::-1])]

            self.check_orientations(e1, orientations)

    def test_transpose(self):
        # tests transposing axes - i.e. changing vantage point on current volume
        for fp in self.filepaths:
            e1 = self.nr.load(fp)
            o = e1.orientation

            orientations = [(o[1], o[2], o[0]), (o[2], o[0], o[1]), (o[1], o[0], o[2]),
                            (o[2], o[1], o[0]), (o[0], o[2], o[1])]

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


if __name__ == '__main__':
    unittest.main()
