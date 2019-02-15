SUPPORTED_FORMATS = ('nifti', 'dicom')


class DataReader():
    """
    This is the class for reading in medical data in various formats
    Currently only 3D data is supported

    Current supported formats:
        DICOM: .dcm
        NIfTi: .nii.gz

    Orientation Conventions:
        - Left: corresponds to patient (not observer) left, RIGHT: corresponds to patient (not observer) right
    """


class DataWriter():
    pass
