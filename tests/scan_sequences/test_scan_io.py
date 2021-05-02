import os

import numpy as np
import pydicom
from pydicom.data import get_testdata_file

from dosma.scan_sequences.scan_io import ScanIOMixin

from .. import util as ututils


class MockScanIOMixin(ScanIOMixin):
    """Mock class for the :cls:`ScanIOMixin`.

    This mixin will be used to load an MR slice hosted on pydicom: ``MR_small.dcm``.
    """

    NAME = "mock-scan-io"
    __DEFAULT_SPLIT_BY__ = None

    def __init__(self, volumes, foo="foo", bar="bar") -> None:
        self.volumes = volumes
        self._from_file_args = {}

        self.foo = foo
        self._bar = bar

        # Attributes that should not be serialized
        self._temp_path = "some/path"
        self.__some_attr__ = 1234
        self.__pydicom_header__ = pydicom.Dataset()

    @property
    def some_property(self):
        return "new/path"


class TestScanIOMixin(ututils.TempPathMixin):
    def test_from_dicom(self):
        mr_dcm = get_testdata_file("MR_small.dcm")
        fs = pydicom.read_file(mr_dcm)
        arr = fs.pixel_array

        scan = MockScanIOMixin.from_dicom(mr_dcm, foo="foofoo", bar="barbar")
        assert len(scan.volumes) == 1
        assert np.all(scan.volumes[0] == arr[..., np.newaxis])
        assert scan.foo == "foofoo"
        assert scan._bar == "barbar"
        assert scan._from_file_args == {
            "dir_or_files": mr_dcm,
            "ignore_ext": False,
            "group_by": None,
            "_type": "dicom",
        }

        scan = MockScanIOMixin.from_dicom([mr_dcm], foo="foofoo", bar="barbar")
        assert len(scan.volumes) == 1
        assert np.all(scan.volumes[0] == arr[..., np.newaxis])
        assert scan.foo == "foofoo"
        assert scan._bar == "barbar"
        assert scan._from_file_args == {
            "dir_or_files": [mr_dcm],
            "ignore_ext": False,
            "group_by": None,
            "_type": "dicom",
        }

    def test_from_dict(self):
        mr_dcm = get_testdata_file("MR_small.dcm")
        scan1 = MockScanIOMixin.from_dicom(mr_dcm)

        scan2 = MockScanIOMixin.from_dict(scan1.__dict__)
        assert scan1.__dict__.keys() == scan2.__dict__.keys()
        for k in scan1.__dict__.keys():
            assert scan1.__dict__[k] == scan2.__dict__[k]

        new_dict = dict(scan1.__dict__)
        new_dict["extra_bool_field"] = True
        with self.assertWarns(UserWarning):
            scan2 = MockScanIOMixin.from_dict(new_dict)
        assert not hasattr(scan2, "extra_bool_field")

        scan2 = MockScanIOMixin.from_dict(new_dict, force=True)
        assert hasattr(scan2, "extra_bool_field")

    def test_save_load(self):
        mr_dcm = get_testdata_file("MR_small.dcm")
        scan = MockScanIOMixin.from_dicom(mr_dcm)
        scan.foo = "foofoo"
        scan._bar = "barbar"

        vars = scan.__serializable_variables__()
        serializable = ("foo", "_bar", "volumes", "_from_file_args")
        not_serializable = ("_temp_path", "__some_attr__", "__pydicom_header__", "some_property")
        assert all(x in vars for x in serializable)
        assert all(x not in vars for x in not_serializable)

        save_dir = os.path.join(self.data_dirpath, "test_save_load")
        save_path = scan.save(save_dir, save_custom=True)
        assert os.path.isfile(save_path)

        scan_loaded = MockScanIOMixin.load(save_path)
        assert scan_loaded.volumes[0].is_identical(scan.volumes[0])
        assert scan_loaded.foo == "foofoo"
        assert scan._bar == "barbar"

        scan_loaded = MockScanIOMixin.load(save_dir)
        assert scan_loaded.volumes[0].is_identical(scan.volumes[0])
        assert scan_loaded.foo == "foofoo"
        assert scan._bar == "barbar"

        with self.assertRaises(FileNotFoundError):
            _ = MockScanIOMixin.load(os.path.join(save_dir, "some-path.pik"))

        new_dict = dict(scan.__dict__)
        new_dict.pop("volumes")
        with self.assertWarns(UserWarning):
            scan_loaded = MockScanIOMixin.load(new_dict)
        assert scan_loaded.volumes[0].is_identical(scan.volumes[0])
        assert scan_loaded.foo == "foofoo"
        assert scan._bar == "barbar"

        # Backwards compatibility with how DOSMA wrote files versions<0.0.12
        new_dict = dict(scan.__dict__)
        new_dict.pop("volumes")
        new_dict.pop("_from_file_args")
        new_dict.update({"dicom_path": mr_dcm, "ignore_ext": False, "series_number": 7})
        with self.assertWarns(UserWarning):
            scan_loaded = MockScanIOMixin.load(new_dict)
        assert scan_loaded.volumes[0].is_identical(scan.volumes[0])
        assert scan_loaded.foo == "foofoo"
        assert scan._bar == "barbar"

        new_dict = dict(scan.__dict__)
        new_dict.pop("volumes")
        new_dict.pop("_from_file_args")
        with self.assertRaises(ValueError):
            _ = MockScanIOMixin.load(new_dict)

        save_dir = os.path.join(self.data_dirpath, "test_save_data")
        with self.assertWarns(DeprecationWarning):
            scan.save_data(save_dir)
