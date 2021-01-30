import difflib
import os
import random
import re
import unittest

import numpy as np
import pydicom

from dosma.data_io.dicom_io import DicomReader, DicomWriter
from dosma.data_io.format_io import ImageDataFormat
from dosma.data_io.nifti_io import NiftiReader, NiftiWriter

from .. import util as ututils


class TestNiftiIO(unittest.TestCase):
    nr = NiftiReader()
    nw = NiftiWriter()

    data_format = ImageDataFormat.nifti

    def test_nifti_read(self):
        for dp in ututils.SCAN_DIRPATHS:
            dicoms_path = ututils.get_dicoms_path(dp)
            read_filepaths = ututils.get_read_paths(dp, self.data_format)

            for read_filepath in read_filepaths:
                _ = self.nr.load(read_filepath)

                with self.assertRaises(FileNotFoundError):
                    _ = self.nr.load(os.path.join(dp, "bleh"))

                with self.assertRaises(FileNotFoundError):
                    _ = self.nr.load(dp)

                with self.assertRaises(ValueError):
                    _ = self.nr.load(os.path.join(dicoms_path, "I0002.dcm"))

    def test_nifti_write(self):
        for dp in ututils.SCAN_DIRPATHS:
            read_filepaths = ututils.get_read_paths(dp, self.data_format)
            save_dirpath = ututils.get_write_path(dp, self.data_format)

            for rfp in read_filepaths:
                save_filepath = os.path.join(save_dirpath, os.path.basename(rfp))
                mv = self.nr.load(rfp)
                self.nw.save(mv, save_filepath)

                # cannot save with extensions other than nii or nii.gz
                with self.assertRaises(ValueError):
                    self.nw.save(mv, os.path.join(ututils.TEMP_PATH, "eg.dcm"))


class TestDicomIO(unittest.TestCase):
    dr = DicomReader()
    dw = DicomWriter()

    data_format = ImageDataFormat.dicom

    @staticmethod
    def are_equivalent_headers(h1, h2):
        """
        Check if two headers are identical. Adapted from
        https://pydicom.github.io/pydicom/stable/auto_examples/plot_dicom_difference.html
        """
        # try:
        #     str(h1), str(h2)
        # except:
        #     return True

        rep = []
        for dataset in (h1, h2):
            lines = str(dataset).split("\n")
            lines = [line + "\n" for line in lines]  # add the newline to end
            rep.append(lines)

        diff = difflib.Differ()
        diff_lines = diff.compare(rep[0], rep[1])
        changes = [
            _line for _line in diff_lines if _line.startswith("+ ") or _line.startswith("- ")
        ]
        return len(changes) == 0

    def test_dicom_reader_total(self):
        for ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]

            dicom_path = ututils.get_dicoms_path(dp)
            volumes = self.dr.load(dicom_path)

            assert type(volumes) is list, "Expected list output"
            expected_num_echos = curr_scan_info["expected_num_echos"]
            assert len(volumes) == expected_num_echos, "Expected %d volumes in %s, got %d" % (
                expected_num_echos,
                curr_scan,
                len(volumes),
            )

            # If multiple echos, all echo volumes must share the same
            # orientation, shape, scanner origin, and pixel spacing.
            e1 = volumes[0]
            for echo_volume in volumes[1:]:
                assert e1.orientation == echo_volume.orientation, (
                    "Orientation mismatch: orientations of multiple dicom volumes "
                    "loaded from single folder should be identical"
                )

                assert (
                    e1.volume.shape == echo_volume.volume.shape
                ), "Shape mismatch: Both volumes should have same shape"

                assert (
                    e1.scanner_origin == echo_volume.scanner_origin
                ), "Scanner origin mismatch: Both volumes should have same scanner_origin"

                assert (
                    e1.pixel_spacing == echo_volume.pixel_spacing
                ), "Pixel spacing mismatch: Both volumes must have the same pixel spacing"

            # Headers for e1 should all have EchoNumbers field = 1
            for i in range(len(volumes)):
                for h in volumes[i].headers:
                    assert h.EchoNumbers == i + 1, "e%d headers should have all EchoNumbers=%d" % (
                        i + 1,
                        i + 1,
                    )

    def test_dicom_reader_separate(self):
        # User manually separates two echos into different folders
        # still be able to read volume in.

        for ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[ind]  # noqa: F841

            multi_echo_read_paths = ututils.get_read_paths(dp, self.data_format)

            dicom_path = ututils.get_dicoms_path(dp)
            volumes = self.dr.load(dicom_path)

            for rfp in multi_echo_read_paths:
                echo_volume = self.dr.load(rfp)
                echo_name = os.path.basename(rfp)
                echo_number = int(re.match(".*?([0-9]+)$", echo_name).group(1))

                assert type(echo_volume) is list, "Output type should be list with length 1"
                assert len(echo_volume) == 1, "length of list must be 1"

                for h in echo_volume[0].headers:
                    assert (
                        h.EchoNumbers == echo_number
                    ), "%s headers should have all EchoNumbers=%d" % (echo_name, echo_number)

                # compare volumes to those read from total - should be identical
                # e1-split (e1): echo1 loaded from the manually separated dicoms
                #   into respective echo folders
                # e1-total (e1_t): echo1 loaded from the total (unseparated) dicoms
                echo_volume = echo_volume[0]
                e_t = volumes[echo_number - 1]

                assert echo_volume.is_identical(
                    e_t
                ), "e%d-split and e%d-total should be identical" % (echo_number, echo_number)

                # headers should also be identical
                assert len(echo_volume.headers) == len(e_t.headers), (
                    "number of headers should be identical in echo %d" % echo_number
                )

                for i in range(len(echo_volume.headers)):
                    h1 = echo_volume.headers[i]
                    h2 = e_t.headers[i]

                    assert self.are_equivalent_headers(h1, h2), (
                        "headers for echos %d must be equivalent" % echo_number
                    )

    def test_dicom_reader_files(self):
        """Test reading dicoms provided as list of files."""
        for ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[ind]  # noqa: F841
            multi_echo_read_paths = ututils.get_read_paths(dp, self.data_format)  # noqa: F841
            dicom_path = ututils.get_dicoms_path(dp)
            expected = self.dr.load(dicom_path)
            volumes = self.dr.load(
                [
                    os.path.join(dicom_path, x)
                    for x in os.listdir(dicom_path)
                    if not x.startswith(".") and x.endswith(".dcm")
                ]
            )

            assert len(expected) == len(volumes)
            for v, e in zip(volumes, expected):
                assert v.is_identical(e)
                assert all(
                    self.are_equivalent_headers(h1, h2) for h1, h2 in zip(v.headers, e.headers)
                )

    def test_dicom_reader_single_file(self):
        """Test reading single dicom file."""
        dp = ututils.SCAN_DIRPATHS[0]
        dicom_path = ututils.get_read_paths(dp, self.data_format)[0]
        dcm_file = random.choice(
            [
                os.path.join(dicom_path, x)
                for x in os.listdir(dicom_path)
                if not x.startswith(".") and x.endswith(".dcm")
            ]
        )
        expected = pydicom.read_file(dcm_file, force=True)
        vol = self.dr.load(dcm_file)[0]

        assert vol.volume.ndim == 3
        assert vol.volume.shape == expected.pixel_array.shape + (1,)

        spacing_expected = [
            expected.PixelSpacing[0],
            expected.PixelSpacing[1],
            expected.SliceThickness,
        ]
        spacing = [np.linalg.norm(vol.affine[:3, i]) for i in range(3)]
        assert np.allclose(spacing, spacing_expected), f"{spacing} == {spacing_expected}"

    def test_dicom_writer(self):
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]  # noqa

            dicom_path = ututils.get_dicoms_path(dp)
            write_path = ututils.get_write_path(dp, self.data_format)
            read_filepaths = ututils.get_read_paths(dp, self.data_format)

            # Read in dicom information, write out to different folder
            # (as multiple echos if possible).
            # Compare to baseline echos separated manually
            volumes = self.dr.load(dicom_path)
            for ind, vol in enumerate(volumes):
                write_fp = os.path.join(write_path, "e%d" % (ind + 1))
                self.dw.save(vol, write_fp)

            for ind, rfp in enumerate(read_filepaths):
                echo_volume_loaded = self.dr.load(rfp)[0]
                e_t = volumes[ind]
                assert echo_volume_loaded.is_identical(
                    e_t
                ), "Loaded e1 and original e1 should be identical"

                # headers should also be identical
                assert len(echo_volume_loaded.headers) == len(
                    e_t.headers
                ), "number of headers should be identical in echo%d" % (ind + 1)

                for i in range(len(echo_volume_loaded.headers)):
                    h1 = echo_volume_loaded.headers[i]
                    h2 = e_t.headers[i]

                    assert self.are_equivalent_headers(
                        h1, h2
                    ), "headers for echoes %d must be equivalent" % (ind + 1)

    def test_dicom_writer_orientation(self):
        # Read in dicom information, reorient image, write out to different folder,
        # compare with multi-echo dicoms
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]  # noqa: F841

            dicom_path = ututils.get_dicoms_path(dp)
            write_path = ututils.get_write_path(dp, self.data_format)

            volumes = self.dr.load(dicom_path)

            # Reorient images
            # echo 1 and echo 2 have the same orientation
            # because they are loaded from the total dicom path
            o = volumes[0].orientation

            # generate random permutations of orientations
            orientations = []
            for i in range(len(volumes)):
                o_new = list(o)
                random.shuffle(o_new)
                o_new = tuple(o_new)
                orientations.append(o_new)

                volumes[i].reformat(o_new, inplace=True)
                self.dw.save(volumes[i], os.path.join(write_path, "e%d" % (i + 1)))

                # orientations should be preserved after saving
                assert (
                    volumes[i].orientation == o_new
                ), "Orientation of echo1 should not change after saving"

            # Currently saving dicom flipped axis (i.e. changing scanner origin)
            # is not supported.
            o3 = (o[0][::-1], o[1], o[2])
            volumes[0].reformat(o3, inplace=True)
            with self.assertRaises(ValueError):
                self.dw.save(volumes[0], ututils.TEMP_PATH)

            # reformat with dicom-specific orientation (o) to compare to loaded echos
            for vol in volumes:
                vol.reformat(o, inplace=True)

            # Load echos from write (save) path
            load_echos_paths = [
                os.path.join(write_path, "e%d" % (i + 1)) for i in range(len(volumes))
            ]

            for ind, rfp in enumerate(load_echos_paths):
                e_loaded = self.dr.load(rfp)[0]
                e_vol = volumes[ind]
                echo_num = ind + 1

                assert e_loaded.is_identical(
                    e_vol
                ), "Loaded e%d and original e%d should be identical" % (echo_num, echo_num)

                # headers should also be identical
                assert len(e_loaded.headers) == len(e_vol.headers), (
                    "number of headers should be identical in echo%d" % echo_num
                )

                for i in range(len(e_loaded.headers)):
                    h1 = e_loaded.headers[i]
                    h2 = e_vol.headers[i]

                    assert self.are_equivalent_headers(h1, h2), (
                        "headers for echoes %d must be equivalent" % echo_num
                    )

    def test_read_multiple_workers(self):
        """Test reading/writing from multiple workers."""
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]  # noqa

            dicom_path = ututils.get_dicoms_path(dp)
            volumes_exp = self.dr.load(dicom_path)
            volumes = DicomReader(num_workers=ututils.num_workers()).load(dicom_path)
            assert len(volumes_exp) == len(volumes)

            for vol, exp in zip(volumes, volumes_exp):
                assert vol.is_identical(exp)

    def test_write_multiple_workers(self):
        """Test reading/writing from multiple workers."""
        for dp_ind, dp in enumerate(ututils.SCAN_DIRPATHS):
            curr_scan = ututils.SCANS[dp_ind]
            curr_scan_info = ututils.SCANS_INFO[curr_scan]  # noqa

            dicom_path = ututils.get_dicoms_path(dp)
            write_path = ututils.get_write_path(dp, self.data_format)
            volumes = self.dr.load(dicom_path)

            for ind, vol in enumerate(volumes):
                exp_path = os.path.join(write_path, "expected", "e%d" % (ind + 1))
                out_path = os.path.join(write_path, "out", "e%d" % (ind + 1))

                self.dw.save(vol, exp_path)
                DicomWriter(num_workers=ututils.num_workers()).save(vol, out_path)

                expected = self.dr.load(exp_path)[0]
                out = self.dr.load(out_path)[0]

                assert out.is_identical(expected)


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
