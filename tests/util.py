import datetime
import os
import re
import shutil
import tempfile
import unittest
import uuid

import natsort
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset

from dosma.cli import SUPPORTED_SCAN_TYPES, parse_args
from dosma.data_io.format_io import ImageDataFormat
from dosma.utils import env, io_utils

UNITTEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "../unittest-data/")
UNITTEST_SCANDATA_PATH = os.path.join(UNITTEST_DATA_PATH, "scans")
TEMP_PATH = os.path.join(
    UNITTEST_SCANDATA_PATH, "temp"
)  # should be used when for writing with assert_raises clauses

SCANS = ["qdess", "mapss", "cubequant", "cones"]
SCANS_INFO = {
    "mapss": {"expected_num_echos": 7},
    "qdess": {"expected_num_echos": 2},
    "cubequant": {"expected_num_echos": 4},
    "cones": {"expected_num_echos": 4},
}

SCAN_DIRPATHS = [os.path.join(UNITTEST_SCANDATA_PATH, x) for x in SCANS]

# Decimal precision for analysis (quantitative values, etc)
DECIMAL_PRECISION = 1  # (+/- 0.1ms)


def build_dummy_headers(shape, fields=None):
    """Build dummy ``pydicom.FileDataset`` headers.

    Note these headers are not dicom compliant and should not be used to write out DICOM
    files.

    Args:
        shape (int or tuple[int]): Shape of headers array.
        fields (Dict): Fields and corresponding values to use to populate the header.

    Returns:
        ndarray: Headers
    """
    if isinstance(shape, int):
        shape = (shape,)
    num_headers = np.prod(shape)
    headers = np.asarray([_build_dummy_pydicom_header(fields) for _ in range(num_headers)])
    return headers.reshape(shape)


def get_scan_dirpath(scan: str):
    for ind, x in enumerate(SCANS):
        if scan == x:
            return SCAN_DIRPATHS[ind]


def get_dicoms_path(fp):
    return os.path.join(fp, "dicoms")


def get_write_path(fp, data_format: ImageDataFormat):
    return os.path.join(fp, "multi-echo-write-%s" % data_format.name)


def get_read_paths(fp, data_format: ImageDataFormat):
    """Get ground truth data (produced by imageviewer like itksnap, horos, etc)"""
    base_name = os.path.join(fp, "multi-echo-gt-%s" % data_format.name)
    files_or_dirs = os.listdir(base_name)
    fd = [x for x in files_or_dirs if re.match("e[0-9]+", x)]
    files_or_dirs = natsort.natsorted(fd)

    return [os.path.join(base_name, x) for x in files_or_dirs]


def get_data_path(fp):
    return os.path.join(fp, f"data-{str(uuid.uuid1())}")


def get_expected_data_path(fp):
    return os.path.join(fp, "expected")


def num_workers() -> int:
    return min(8, os.cpu_count())


def requires_packages(*packages):
    """
    Decorator for functions that should only execute when
    all packages defined by *args are supported.
    """

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            if all([env._package_available(x) for x in packages]):
                func(*args, **kwargs)

        return _wrapper

    return _decorator


def _build_dummy_pydicom_header(fields=None):
    """Builds dummy pydicom-based header.

    Note these headers are not dicom compliant and should not be used to write out DICOM
    files.

    Adapted from
    https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_write_dicom.html
    """
    suffix = ".dcm"
    filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    # Create the FileDataset instance (initially no data elements, but file_meta supplied).
    ds = FileDataset(filename_little_endian, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add the data elements -- not trying to set all required here. Check DICOM standard.
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    if fields is not None:
        for k, v in fields.items():
            setattr(ds, k, v)

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    timeStr = dt.strftime("%H%M%S.%f")  # long format with micro seconds
    ds.ContentTime = timeStr

    return ds


class ScanTest(unittest.TestCase):
    from dosma.scan_sequences.scans import ScanSequence

    SCAN_TYPE = ScanSequence  # override in subclasses

    dicom_dirpath = None
    data_dirpath = None

    def setUp(self):
        print("Testing: ", self._testMethodName)

    @classmethod
    def setUpClass(cls):
        cls.dicom_dirpath = get_dicoms_path(
            os.path.join(UNITTEST_SCANDATA_PATH, cls.SCAN_TYPE.NAME)
        )
        cls.data_dirpath = get_data_path(os.path.join(UNITTEST_SCANDATA_PATH, cls.SCAN_TYPE.NAME))
        io_utils.mkdirs(cls.data_dirpath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dirpath)

    def test_has_cmd_line_actions_attr(self):
        """
        If scan can be accessed via the command line,
        verify that the scan has a ``cmd_line_actions`` method.
        """
        # if the scan is not supported via the command line, then ignore this test
        if self.SCAN_TYPE not in SUPPORTED_SCAN_TYPES:
            return

        assert hasattr(
            self.SCAN_TYPE, "cmd_line_actions"
        ), "All scans supported by command line must have `cmd_line_actions` method"
        assert hasattr(
            type(self), "test_cmd_line"
        ), "All scan supported in command line must have test methods `test_cmd_line`"

    def __cmd_line_helper__(self, cmdline_str: str):
        cmdline_input = cmdline_str.strip().split()
        parse_args(cmdline_input)
