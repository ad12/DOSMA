import unittest


class TestNiftiIO(unittest.TestCase):
    from data_io.nifti_io import NiftiReader, NiftiWriter

    load_filepath = '../dicoms/healthy07/h7.nii.gz'
    save_filepath = '../dicoms/healthy07/h7_nifti_writer.nii.gz'

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
    pass


class TestInterIo(unittest.TestCase):
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
        assert o == e2_dcm.orientation, "orientations of multiple dicom volumes loaded from single call should be identical"

        e1_itksnap.reformat(o)
        e2_itksnap.reformat(o)

        assert (e1_dcm.volume == e1_itksnap.volume).all(), "e1 volumes (dcm, itksnap) should be identical in the same format"
        assert (e2_dcm.volume == e2_itksnap.volume).all(), "e2 volumes (dcm, itksnap) should be identical in the same format"

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
