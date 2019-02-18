import unittest


class TestNiftiIO(unittest.TestCase):
    from data_io.nifti_io import NiftiReader, NiftiWriter

    load_filepath = '../dicoms/h07-unittest/dess-e1-itksnap.nii.gz'
    save_filepath = '../dicoms/h07-unittest/dess-e1-niftiwriter.nii.gz'

    nr = NiftiReader()
    nw = NiftiWriter()

    def test_nifti_read(self):
        mv = self.nr.load(self.load_filepath)

        with self.assertRaises(FileNotFoundError):
            mv = self.nr.load('../dicoms/healthy07/h7')

        with self.assertRaises(FileNotFoundError):
            mv = self.nr.load('../dicoms/healthy07')

        with self.assertRaises(ValueError):
            mv = self.nr.load('../dicoms/healthy07/007/I0001.dcm')

    def test_nifti_write(self):
        mv = self.nr.load(self.load_filepath)
        self.nw.save(mv, self.save_filepath)
        self.nw.save(mv, '../dicoms/healthy07/h7_nifti_writer.nii')

        with self.assertRaises(ValueError):
            self.nw.save(mv, '../dicoms/healthy07/h7_nifti_writer.dcm')


class TestDicomIO(unittest.TestCase):
    from data_io.dicom_io import DicomReader, DicomWriter

    dr = DicomReader()
    dw = DicomWriter()

    # Dicom load path - load both echos and separate using DicomReader
    total_dcm_path = '../dicoms/h07-unittest/007'
    e1_dcm_path = '../dicoms/h07-unittest/dess-e1'
    e2_dcm_path = '../dicoms/h07-unittest/dess-e2'

    # Dicom save paths - echos saved separately
    e1_dcm_save_path = '../dicoms/h07-unittest/dess-e1-dicom_writer'
    e2_dcm_save_path = '../dicoms/h07-unittest/dess-e2-dicom_writer'

    @staticmethod
    def are_equivalent_headers(h1, h2):
        """
        Check if two headers are identical
        Adapted from https://pydicom.github.io/pydicom/stable/auto_examples/plot_dicom_difference.html
        :param h1:
        :param h2:
        :return:
        """
        import difflib

        rep = []
        for dataset in (h1, h2):
            lines = str(dataset).split("\n")
            lines = [line + "\n" for line in lines]  # add the newline to end
            rep.append(lines)

        diff = difflib.Differ()
        diff_lines = diff.compare(rep[0], rep[1])
        changes = [l for l in diff_lines if l.startswith('+ ') or l.startswith('- ')]
        return len(changes) == 0

    def test_dicom_reader_total(self):
        # Dicom reader should read in both volumes separately for qDESS series
        volumes = self.dr.load(self.total_dcm_path)

        assert type(volumes) is list, "Expected list output"
        assert len(volumes) == 2, "Expected 2 volumes in qDESS, got %d" % len(volumes)

        # echo1 (e1) and echo2 (e2) should have same shape
        e1 = volumes[0]
        e2 = volumes[1]

        # Get dicom and itksnap in same orientation
        o = e1.orientation
        assert o == e2.orientation, "orientations of multiple dicom volumes loaded from single folder should be identical"

        assert e1.volume.shape == e2.volume.shape, "Both volumes should have same shape"

        # Headers for e1 should all have EchoNumbers field = 1
        for i in range(len(volumes)):
            for h in volumes[i].headers:
                assert h.EchoNumbers == i + 1, "e%d headers should have all EchoNumbers=%d" % (i + 1, i + 1)

    def test_dicom_reader_separate(self):
        # User manually separates two echos into different folders
        # still be able to read volume in

        e1 = self.dr.load(self.e1_dcm_path)
        assert type(e1) is list, "Output type should be list with length 1"
        assert len(e1) == 1, "length of list must be 1"

        for h in e1[0].headers:
            assert h.EchoNumbers == 1, "e1 headers should have all EchoNumbers=1"

        e2 = self.dr.load(self.e2_dcm_path)
        assert type(e2) is list, "Output type should be list with length 1"
        assert len(e2) == 1, "length of list must be 1"

        for h in e2[0].headers:
            assert h.EchoNumbers == 2, "e2 headers should have all EchoNumbers=2"

        # single element in a list, unpack it for easy access
        e1 = e1[0]
        e2 = e2[0]

        # compare volumes to those read from total - should be identical
        # e1-split (e1): echo1 loaded from the manually separated dicoms into respective echo folders
        # e1-total (e1_t): echo1 loaded from the total (unseparated) dicoms
        # similarly for e2 (echo2)
        volumes = self.dr.load(self.total_dcm_path)
        e1_t = volumes[0]
        e2_t = volumes[1]

        assert e1.is_identical(e1_t), "e1-split and e1-total should be identical"
        assert e2.is_identical(e2_t), "e2-split and e2-total should be identical"

        count = 0
        # headers should also be identical
        for e, et in [(e1, e1_t), (e2, e2_t)]:
            count += 1
            assert len(e.headers) == len(et.headers), "number of headers should be identical in echo%d" % count

            for i in range(len(e.headers)):
                h1 = e.headers[i]
                h2 = et.headers[i]

                assert self.are_equivalent_headers(h1, h2), "headers for echoes %d must be equivalent" % count

    def test_dicom_writer(self):
        # Read in dicom information, write out to different folder, compare with e1 and e2 dicoms
        volumes = self.dr.load(self.total_dcm_path)
        e1 = volumes[0]
        e2 = volumes[1]

        self.dw.save(e1, self.e1_dcm_save_path)
        self.dw.save(e2, self.e2_dcm_save_path)

        e1_loaded = self.dr.load(self.e1_dcm_save_path)
        e2_loaded = self.dr.load(self.e2_dcm_save_path)

        e1_loaded = e1_loaded[0]
        e2_loaded = e2_loaded[0]

        assert e1_loaded.is_identical(e1), "Loaded e1 and original e1 should be identical"
        assert e2_loaded.is_identical(e2), "Loaded e2 and original e2 should be identical"

        count = 0
        # headers should also be identical
        for e, et in [(e1_loaded, e1), (e2_loaded, e2)]:
            count += 1
            assert len(e.headers) == len(et.headers), "number of headers should be identical in echo%d" % count

            for i in range(len(e.headers)):
                h1 = e.headers[i]
                h2 = et.headers[i]

                assert self.are_equivalent_headers(h1, h2), "headers for echoes %d must be equivalent" % count

    def test_dicom_writer_orientation(self):
        # Read in dicom information, reorient image, write out to different folder, compare with e1 and e2 dicoms
        volumes = self.dr.load(self.total_dcm_path)
        e1 = volumes[0]
        e2 = volumes[1]

        # Reorient images
        o = e1.orientation  # echo 1 and echo 2 have the same orientation, because loaded from the total dicom path

        o1 = (o[1], o[2], o[0])
        e1.reformat(o1)

        o2 = (o[2], o[0], o[1])
        e2.reformat(o2)

        # save images
        self.dw.save(e1, self.e1_dcm_save_path)
        self.dw.save(e2, self.e2_dcm_save_path)

        # orientations should be preserved after saving
        assert e1.orientation == o1, "Orientation of echo1 should not change after saving"
        assert e2.orientation == o2, "Orientation of echo2 should not change after saving"

        # Currently saving dicom flipped axis (i.e. changing scanner origin) is not supported
        # check to make sure error is raised
        o3 = (o[0][::-1], o[1], o[2])
        e2.reformat(o3)
        with self.assertRaises(ValueError):
            self.dw.save(e2, self.e2_dcm_save_path)

        # reformat with dicom-specific orientation (o) to compare to loaded echos
        e1.reformat(o)
        e2.reformat(o)

        # Load echos from write path
        e1_loaded = self.dr.load(self.e1_dcm_save_path)
        e2_loaded = self.dr.load(self.e2_dcm_save_path)

        e1_loaded = e1_loaded[0]
        e2_loaded = e2_loaded[0]

        assert e1_loaded.is_identical(e1), "Loaded e1 and original e1 should be identical"
        assert e2_loaded.is_identical(e2), "Loaded e2 and original e2 should be identical"

        count = 0
        # headers should also be identical
        for e, et in [(e1_loaded, e1), (e2_loaded, e2)]:
            count += 1
            assert len(e.headers) == len(et.headers), "number of headers should be identical in echo%d" % count

            for i in range(len(e.headers)):
                h1 = e.headers[i]
                h2 = et.headers[i]

                assert self.are_equivalent_headers(h1, h2), "headers for echoes %d must be equivalent" % count


class TestInterIO(unittest.TestCase):
    from data_io.nifti_io import NiftiReader, NiftiWriter
    from data_io.dicom_io import DicomReader, DicomWriter

    nr = NiftiReader()
    nw = NiftiWriter()

    dr = DicomReader()
    dw = DicomWriter()

    # filepaths to DESS echo1 and echo2 volumes created by ITK-Snap - serve as ground truth
    e1_nifti_itksnap_path = '../dicoms/h07-unittest/dess-e1-itksnap.nii.gz'
    e2_nifti_itksnap_path = '../dicoms/h07-unittest/dess-e2-itksnap.nii.gz'

    # Dicom load path - load both echos and separate using DicomReader
    total_dcm_path = '../dicoms/h07-unittest/007'
    e1_dcm_path = '../dicoms/h07-unittest/dess-e1'
    e2_dcm_path = '../dicoms/h07-unittest/dess-e2'

    # Nifti save path
    e1_nifti_save_path = '../dicoms/h07-unittest/dess-e1-niftiwriter.nii.gz'
    e2_nifti_save_path = '../dicoms/h07-unittest/dess-e2-niftiwriter.nii.gz'

    def test_dcm_to_nifti(self):
        # Load itksnap ground truth
        e1_itksnap = self.nr.load(self.e1_nifti_itksnap_path)
        e2_itksnap = self.nr.load(self.e2_nifti_itksnap_path)

        # DicomReader to read multiple echo volumes from DESS sequence
        vols = self.dr.load(self.total_dcm_path)
        e1_dcm = vols[0]
        e2_dcm = vols[1]

        # Get dicom and itksnap in same orientation
        o = e1_dcm.orientation
        assert o == e2_dcm.orientation, "orientations of multiple dicom volumes loaded from single folder should be identical"

        e1_itksnap.reformat(o)
        e2_itksnap.reformat(o)

        assert (e1_dcm.volume == e1_itksnap.volume).all(), "e1 volumes (dcm, itksnap) should be identical"
        assert (e2_dcm.volume == e2_itksnap.volume).all(), "e2 volumes (dcm, itksnap) should be identical"

        # Use NiftiWriter to save volumes (read in as dicoms)
        self.nw.save(e1_dcm, self.e1_nifti_save_path)
        self.nw.save(e2_dcm, self.e2_nifti_save_path)

        # check if saved versions of volumes load correctly
        e1_nifti = self.nr.load(self.e1_nifti_save_path)
        e2_nifti = self.nr.load(self.e2_nifti_save_path)

        assert e1_nifti.is_same_dimensions(e1_itksnap)
        assert e2_nifti.is_same_dimensions(e2_itksnap)

        assert (e1_dcm.volume == e1_itksnap.volume).all()
        assert (e2_dcm.volume == e2_itksnap.volume).all()


if __name__ == '__main__':
    unittest.main()
