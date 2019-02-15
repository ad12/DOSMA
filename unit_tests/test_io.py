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


if __name__ == '__main__':
    unittest.main()
