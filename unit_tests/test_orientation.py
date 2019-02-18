import unittest


class TestOrientation(unittest.TestCase):
    from data_io.nifti_io import NiftiReader, NiftiWriter
    from data_io.med_volume import MedicalVolume
    nr = NiftiReader()
    nw = NiftiWriter()

    # filepaths to DESS echo1 and echo2 volumes created by ITK-Snap - serve as ground truth
    e1_nifti_itksnap_path = '../dicoms/h07-unittest/dess-e1-itksnap.nii.gz'
    e2_nifti_itksnap_path = '../dicoms/h07-unittest/dess-e2-itksnap.nii.gz'

    def check_orientations(self, mv: MedicalVolume, orientations):
        """
        Apply each orientation specified in orientations to the Medical Volume mv
        Assert if mv --> apply orientation --> apply original orientation != mv original position coordinates
        :param mv: a Medical Volume
        :param orientations: a list or tuple of orientation tuples
        """
        o_base, so_base, ps_base = mv.orientation, mv.scanner_origin, mv.pixel_spacing

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

    def test_flip(self):
        # tests flipping orientation across volume axis
        e1 = self.nr.load(self.e1_nifti_itksnap_path)
        o = e1.orientation
        orientations = [(o[0][::-1], o[1], o[2]), (o[0], o[1][::-1], o[2]), (o[0], o[1][::-1], o[2]),
                        (o[0][::-1], o[1][::-1], o[2]), (o[0][::-1], o[1], o[2][::-1]), (o[0], o[1][::-1], o[2][::-1]),
                        (o[0][::-1], o[1][::-1], o[2][::-1])]

        self.check_orientations(e1, orientations)

    def test_transpose(self):
        # tests transposing axes - i.e. changing vantage point on current volume
        e1 = self.nr.load(self.e1_nifti_itksnap_path)
        o = e1.orientation

        orientations = [(o[1], o[2], o[0]), (o[2], o[0], o[1]), (o[1], o[0], o[2]),
                        (o[2], o[1], o[0]), (o[0], o[2], o[1])]

        self.check_orientations(e1, orientations)

    def test_transpose_and_flip(self):
        from itertools import permutations
        e1 = self.nr.load(self.e1_nifti_itksnap_path)
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
