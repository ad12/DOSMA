from data_io.dicom_io import DicomWriter
from data_io.format_io import SUPPORTED_FORMATS
from data_io.nifti_io import NiftiWriter


def get_writer(data_format: str):
    if format not in SUPPORTED_FORMATS:
        raise ValueError('Only formats %s are supported' % str(SUPPORTED_FORMATS))

    if data_format == 'nifti':
        return NiftiWriter()
    elif data_format == 'dicom':
        return DicomWriter()