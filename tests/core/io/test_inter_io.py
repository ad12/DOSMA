import os
import unittest

from dosma.core.io.dicom_io import DicomReader, DicomWriter
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter

from ... import util as ututils


class TestInterIO(unittest.TestCase):
    nr = NiftiReader()
    nw = NiftiWriter()

    dr = DicomReader()
    dw = DicomWriter()

    @staticmethod
    def compare_vols(vol1, vol2):
        assert vol1.is_same_dimensions(vol2)
        assert (vol1.volume == vol2.volume).all()

    def test_dcm_nifti_load(self):
        """Verify that volumes loaded from nifti or dicom are identical"""
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]  # noqa

            nifti_filepaths = ututils.get_read_paths(dp, ImageDataFormat.nifti)
            dicom_filepaths = ututils.get_read_paths(dp, ImageDataFormat.dicom)

            for i in range(len(nifti_filepaths)):
                nfp = nifti_filepaths[i]
                dfp = dicom_filepaths[i]

                nifti_vol = self.nr.load(nfp)
                dicom_vol = self.dr.load(dfp)[0]
                dicom_vol.reformat(nifti_vol.orientation, inplace=True)

                # assert nifti_vol.is_same_dimensions(dicom_vol)
                assert (nifti_vol.volume == dicom_vol.volume).all()

    def test_dcm_to_nifti(self):
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]  # noqa

            dicom_path = ututils.get_dicoms_path(dp)
            nifti_read_paths = ututils.get_read_paths(dp, ImageDataFormat.nifti)
            nifti_write_path = ututils.get_write_path(dp, ImageDataFormat.nifti)

            # Load ground truth (nifti format)
            gt_nifti_vols = []
            for rfp in nifti_read_paths:
                gt_nifti_vols.append(self.nr.load(rfp))

            # DicomReader to read multiple echo volumes from scan sequence.
            dicom_loaded_vols = self.dr.load(dicom_path)

            # Get dicom and itksnap in same orientation
            o = dicom_loaded_vols[0].orientation
            for v in dicom_loaded_vols[1:]:
                assert o == v.orientation, (
                    "Orientations of multiple dicom volumes loaded from single folder "
                    "should be identical"
                )

            for v in gt_nifti_vols:
                v.reformat(o, inplace=True)

            for i in range(len(dicom_loaded_vols)):
                dcm_vol = dicom_loaded_vols[i]
                nifti_vol = gt_nifti_vols[i]
                echo_num = i + 1

                assert (dcm_vol.volume == nifti_vol.volume).all(), (
                    "e%d volumes (dcm, nifti-ground truth) should be identical" % echo_num
                )

            # Use NiftiWriter to save volumes (read in as dicoms)
            for i in range(len(dicom_loaded_vols)):
                dcm_vol = dicom_loaded_vols[i]
                echo_num = i + 1

                nifti_write_filepath = os.path.join(nifti_write_path, "e%d.nii.gz" % echo_num)
                self.nw.save(dcm_vol, nifti_write_filepath)

                # check if saved versions of volumes load correctly
                e_loaded = self.nr.load(nifti_write_filepath)
                e_dcm = dcm_vol
                e_gt_nifti = gt_nifti_vols[i]

                # assert e_loaded.is_same_dimensions(e_gt_nifti), (
                #     "Mismatched dimensions: %s echo-%d" % (curr_scan, i+1)
                # )

                assert (e_dcm.volume == e_gt_nifti.volume).all()
                assert (e_loaded.volume == e_gt_nifti.volume).all()


if __name__ == "__main__":
    unittest.main()
