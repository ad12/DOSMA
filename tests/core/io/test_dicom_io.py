import difflib
import os
import random
import re
import unittest

import numpy as np
import pydicom
from pydicom.data import get_testdata_file

from dosma.core.io.dicom_io import DicomReader, DicomWriter, to_RAS_affine
from dosma.core.io.format_io import ImageDataFormat
from dosma.core.med_volume import MedicalVolume

from ... import util as ututils


class TestDicomIO(ututils.TempPathMixin):
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

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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
                for h in volumes[i].headers(flatten=True):
                    assert h.EchoNumbers == i + 1, "e%d headers should have all EchoNumbers=%d" % (
                        i + 1,
                        i + 1,
                    )

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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

                for h in echo_volume[0].headers(flatten=True):
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
                ev_headers = echo_volume.headers(flatten=True)
                e_t_headers = e_t.headers(flatten=True)
                assert len(ev_headers) == len(e_t_headers), (
                    "number of headers should be identical in echo %d" % echo_number
                )

                for i in range(len(ev_headers)):
                    h1 = ev_headers[i]
                    h2 = e_t_headers[i]

                    assert self.are_equivalent_headers(h1, h2), (
                        "headers for echos %d must be equivalent" % echo_number
                    )

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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
                    self.are_equivalent_headers(h1, h2)
                    for h1, h2 in zip(v.headers(flatten=True), e.headers(flatten=True))
                )

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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
                evl_headers = echo_volume_loaded.headers(flatten=True)
                e_t_headers = e_t.headers(flatten=True)
                assert len(evl_headers) == len(
                    e_t_headers
                ), "number of headers should be identical in echo%d" % (ind + 1)

                for i in range(len(evl_headers)):
                    h1 = evl_headers[i]
                    h2 = e_t_headers[i]

                    assert self.are_equivalent_headers(
                        h1, h2
                    ), "headers for echoes %d must be equivalent" % (ind + 1)

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_dicom_writer_nd(self):
        """Test writing dicoms for >3D MedicalVolume data."""
        dicom_path = ututils.get_dicoms_path(ututils.get_scan_dirpath("qdess"))
        e1, e2 = tuple(self.dr.load(dicom_path))
        vol, headers = np.stack([e1, e2], axis=-1), np.stack([e1.headers(), e2.headers()], axis=-1)
        vol = MedicalVolume(vol, affine=e1.affine, headers=headers)

        write_path = ututils.get_write_path(dicom_path, self.data_format)
        self.dw.save(vol, write_path)

        e1_l, e2_l = tuple(self.dr.load(write_path))
        assert e1_l.is_identical(e1)
        assert e2_l.is_identical(e2)

        for ind, (e_l, e) in enumerate([(e1_l, e1), (e2_l, e2)]):
            el_headers = e_l.headers(flatten=True)
            e_headers = e.headers(flatten=True)
            for i in range(len(el_headers)):
                h1 = el_headers[i]
                h2 = e_headers[i]
                assert self.are_equivalent_headers(
                    h1, h2
                ), "headers for echoes %d must be equivalent" % (ind + 1)

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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
                el_headers = e_loaded.headers(flatten=True)
                ev_headers = e_vol.headers(flatten=True)
                assert len(el_headers) == len(ev_headers), (
                    "number of headers should be identical in echo%d" % echo_num
                )

                for i in range(len(el_headers)):
                    h1 = el_headers[i]
                    h2 = ev_headers[i]

                    assert self.are_equivalent_headers(h1, h2), (
                        "headers for echoes %d must be equivalent" % echo_num
                    )

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
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

    def test_special_affine(self):
        """Test creation of affine matrix for special cases."""
        # Patient orientation (useful for xray data).
        header = ututils.build_dummy_headers(
            1, fields={"PatientOrientation": ["P", "F"], "PixelSpacing": [0.2, 0.5]}
        )
        affine = to_RAS_affine(header)
        mv = MedicalVolume(np.ones((10, 20, 30)), affine=affine)
        assert mv.orientation == ("SI", "AP", "LR")
        assert mv.pixel_spacing == (0.5, 0.2, 1.0)
        assert mv.scanner_origin == (0.0, 0.0, 0.0)

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_load_sort_by(self):
        """Test sorting by dicom attributes."""
        dp = ututils.SCAN_DIRPATHS[0]
        dicom_path = ututils.get_dicoms_path(dp)
        vols = self.dr.load(dicom_path, sort_by="InstanceNumber")

        for v in vols:
            instance_numbers = [h.InstanceNumber for h in v.headers(flatten=True)]
            assert instance_numbers == sorted(instance_numbers)

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_write_sort_by(self):
        """Test sorting by dicom attributes before writing."""
        dp = ututils.get_scan_dirpath("qdess")
        dicom_path = ututils.get_dicoms_path(dp)
        vols = self.dr.load(dicom_path)
        vol = np.stack(vols)

        write_path = os.path.join(ututils.get_write_path(dp, self.data_format), "out-multi")
        self.dw.save(vol, write_path, sort_by="InstanceNumber")

        files = [os.path.join(write_path, x) for x in sorted(os.listdir(write_path))]
        e1_files, e2_files = files[::2], files[1::2]

        e1_vols, e2_vols = self.dr.load(e1_files), self.dr.load(e2_files)
        assert len(e1_vols) == len(e2_vols) == 1
        e1, e2 = e1_vols[0], e2_vols[0]

        assert e1.is_identical(vols[0])
        assert e2.is_identical(vols[1])

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_load_no_group_by(self):
        """Test reading dicoms without group_by."""
        dp = ututils.get_scan_dirpath("qdess")
        # Echo 1 only
        fp = ututils.get_read_paths(dp, self.data_format)[0]

        dicom_path = ututils.get_dicoms_path(dp)
        e1_expected = self.dr.load(dicom_path)[0]

        e1 = self.dr.load(fp, group_by=None)[0]

        assert e1.is_identical(e1_expected)

    @unittest.skipIf(not ututils.is_data_available(), "unittest data is not available")
    def test_init_params(self):
        """Test reading/writing works with passing values to constructor."""
        dp = ututils.get_scan_dirpath("qdess")
        # Echo 1 only
        fp = ututils.get_read_paths(dp, self.data_format)[0]

        e1_expected = self.dr.load(fp, group_by=None, sort_by="InstanceNumber", ignore_ext=False)[0]
        dr = DicomReader(group_by=None, sort_by="InstanceNumber", ignore_ext=False)
        e1 = dr.load(fp)[0]
        e1_expected.orientation

        assert e1.is_identical(e1_expected)

        write_path = os.path.join(ututils.get_write_path(dp, self.data_format), "out-init-params")
        dw = DicomWriter(fname_fmt="I%05d.dcm", sort_by="InstanceNumber")
        dw.save(e1_expected, write_path, sort_by="InstanceNumber")

        files = [_f for _f in os.listdir(write_path) if _f.endswith(".dcm")]
        assert len(files) == e1_expected.shape[-1]

    def test_sample_pydicom_data(self):
        """Test DICOM reader with sample pydicom data."""
        filepath = get_testdata_file("MR_small.dcm")
        mv_pydicom = pydicom.read_file(filepath)
        arr = mv_pydicom.pixel_array

        dr = DicomReader(group_by=None)
        mv = dr(filepath)
        assert len(mv) == 1
        mv = mv[0]
        assert mv.shape == (arr.shape) + (1,)
        assert self.are_equivalent_headers(mv.headers(flatten=True)[0], mv_pydicom)

        dw = DicomWriter()
        out_dir = os.path.join(ututils.TEMP_PATH, "dicom_sample_pydicom")
        out_path = os.path.join(out_dir, "I0001.dcm")
        dw(mv, dir_path=out_dir)

        mv_pydicom_loaded = pydicom.read_file(out_path)
        assert np.all(mv_pydicom_loaded.pixel_array == arr)
        assert self.are_equivalent_headers(mv_pydicom_loaded, mv_pydicom)

    def test_load_pydicom_data(self):
        filepath = get_testdata_file("MR_small.dcm")
        dr = DicomReader(group_by=None)
        mv = dr.load(filepath)[0]

        # CT does not have EchoNumbers field in header.
        with self.assertRaises(KeyError):
            dr.load(get_testdata_file("CT_small.dcm"), group_by="EchoNumbers")
        with self.assertRaises(KeyError):
            dr.load(get_testdata_file("CT_small.dcm"), sort_by="EchoNumbers")

        # Multiprocessing
        dr_mp = DicomReader(group_by=None, num_workers=1)
        mv2 = dr_mp.load(filepath)[0]
        assert mv2.is_identical(mv)

        dr_mp = DicomReader(group_by=None, num_workers=1, verbose=True)
        mv2 = dr_mp.load(filepath)[0]
        assert mv2.is_identical(mv)

        # sort by
        mv2 = dr.load(filepath, sort_by="InstanceNumber")[0]
        assert mv2.is_identical(mv)

        # bytes
        with open(filepath, "rb") as f:
            mv2 = dr.load(f)[0]
        assert mv2.is_identical(mv)

    def test_save_different_bits(self):
        """Test writing volume where bit depth has changed."""
        filepath = get_testdata_file("MR_small.dcm")
        dr = DicomReader(group_by=None)
        mv_base = dr.load(filepath)[0]

        arr = (np.random.rand(*mv_base.shape) > 0.5).astype(np.uint8) * 255
        mv = mv_base._partial_clone(volume=arr)

        dw = DicomWriter()
        out_dir = os.path.join(self.data_dirpath, "test_write_different_bits")
        dw.save(mv, dir_path=out_dir)

        dr = DicomReader(group_by=None)
        mv2 = dr.load(out_dir)[0]
        assert mv2.is_identical(mv)

    def test_save(self):
        filepath = get_testdata_file("MR_small.dcm")
        dr = DicomReader(group_by=None)
        mv_base = dr.load(filepath)[0]

        out_dir = os.path.join(self.data_dirpath, "test_save_sort_by")
        dw = DicomWriter()
        dw.save(mv_base, out_dir, sort_by="InstanceNumber")
        mv2 = dr.load(out_dir)[0]
        assert mv2.is_identical(mv_base)

        out_dir = os.path.join(self.data_dirpath, "test_save_no_headers")
        mv = MedicalVolume(np.ones((10, 10, 10)), np.eye(4))
        dw = DicomWriter()
        with self.assertRaises(ValueError):
            dw.save(mv, out_dir)

    def test_state(self):
        dr1 = DicomReader()
        state_dict = dr1.state_dict()
        state_dict.update({"num_workers": 8, "group_by": None})

        dr1.num_workers = 5
        dr1.group_by = "foo"

        dr2 = DicomReader()
        state_dict = dr2.load_state_dict(state_dict)
        assert dr2.num_workers == 8
        assert dr2.group_by is None

        dw1 = DicomWriter()
        state_dict = dw1.state_dict()
        state_dict.update({"num_workers": 8, "sort_by": "InstanceNumber"})

        dw2 = DicomWriter()
        state_dict = dw2.load_state_dict(state_dict)
        assert dw2.num_workers == 8
        assert dw2.sort_by == "InstanceNumber"

    def test_get_files(self):
        # Make dummy files with some properties.
        out_dir = self.data_dirpath / "test_get_files"
        os.makedirs(out_dir, exist_ok=True)
        filenames = [
            "I0001.dcm",
            "I0002.dcm",
            "I0003.dcm",
            "I0004.dcm",
            "I0005.dcm",
            "I0006",
            "I0007",
            "I0008",
            "I0009",
            "I0010",
            "I0011-sft.dcm",
            "I0012-sft.dcm",
            "I0013-sft.dcm",
            "I0014-sft.dcm",
            "I0015-sft.dcm",
            ".I0016.dcm",
        ]
        filenames = [out_dir / fname for fname in filenames]
        filenames_str = [str(x) for x in filenames]
        for fname in filenames:
            with open(fname, "w"):
                pass

        dr = DicomReader()
        files = dr.get_files(out_dir, ignore_ext=False, ignore_hidden=False)
        assert set(files) == {x for x in filenames_str if x.endswith(".dcm")}

        dr = DicomReader()
        files = dr.get_files(out_dir, ignore_ext=False, ignore_hidden=True)
        assert set(files) == {
            x
            for x in filenames_str
            if not os.path.basename(x).startswith(".") and x.endswith(".dcm")
        }

        dr = DicomReader()
        files = dr.get_files(out_dir, ignore_ext=True, ignore_hidden=False)
        assert set(files) == set(filenames_str)

        dr = DicomReader()
        files = dr.get_files(out_dir, ignore_ext=True, ignore_hidden=True)
        assert set(files) == {x for x in filenames_str if not os.path.basename(x).startswith(".")}

        with self.assertRaises(NotADirectoryError):
            dr.get_files(out_dir / "some-folder")


if __name__ == "__main__":
    unittest.main()
