.. _basic_usage:

**This guide is still under construction**

Basic Usage
-----------

Dosma is designed for simple imaging I/O, registration, quantitative fitting, and AI-based image processing. 

Dosma does not bundle Tensorflow and Keras installation by default.
To enable  support, you must install these libraries as an additional step.


Image I/O
=========================

Dosma provides data readers and writers to allow you to read/write image data stored in NIfTI and DICOM standards.
These I/O tools create or write from the Dosma image class :class:`MedicalVolume`.

For example to load a DICOM image series, which has multiple echos, with each echo corresponding to a volume,
we can do:

>>> from dosma.data_io import DicomReader:
>>> with DicomReader() as dr:
>>>   volumes = dr.load("/path/to/dicom/folder", group_by="EchoNumbers")

We can also load specific files in the image series:

>>> with DicomReader() as dr:
>>>   volumes = dr.load(["file1", "file2", ...], group_by="EchoNumbers")

DICOM image data often has associated metadata. :class:`MedicalVolume` makes it easy to get
and set metadata:

>>> volume = volumes[0]  # first echo time
>>> volume.get_metadata("EchoTime", float)
10.0
>>> volume.set_metadata("EchoTime", 20)
>>> volume.get_metadata("EchoTime", float)
20.0

Similarly, to load a NIfTI volume, we use the :class:`NiftiReader` class:

>>> from dosma.data_io import NiftiReader
>>> with NiftiReader() as nr:
>>>   volume = nr.load("/path/to/nifti/file")


Reformatting Images
=========================

Given the multiple different orientation conventions used by different image formats and libraries,
reformatting medical images can be difficult to keep track of. Dosma simplifies this by introducing
an unambiguous convention for image orientation based on the RAS+ coordinate system, in which all
directions point to the increasing direction.

To reformat a :class:`MedicalVolume` instance (``mv``) such that the dimensions correspond to
superior -> inferior, anterior -> posterior, left -> right, we can do:

>>> mv = mv.reformat(("SI", "AP", "LR"))

To perform the operation in-place (i.e. modifying the existing instance), we can do:

>>> mv = mv.reformat(("SI", "AP", "LR"), inplace=True)

Note, in-place reformatting returns the same :class:`MedicalVolume` object that was modified
in-place (i.e. ``self``) to allow chaining methods together.

We may also want to reformat images to be in the same orientation as other images:

>>> mv = mv.reformat_as(other_image)


Image Slicing and Arithmetic Operations
========================================

:class:`MedicalVolume` supports some array-like functionality, including Python arithmetic
operations (``+``, ``-``, ``**``, ``/``, ``//``), NumPy shape-preserving operations
(e.g. ``np.exp``, ``np.log``, ``np.pow``, etc.), and slicing.

>>> mv += 5
>>> mv = mv * mv / mv
>>> mv = np.exp(mv)
>>> mv = mv[:5, :6, :7]

Note, in order to preserve dimensions, slicing cannot be used to reduce dimensions.
For example, the first line will throw an error; the second will not:

>>> mv = mv[2]
IndexError: Scalar indices disallowed in spatial dimensions; Use `[x]` or `x:x+1`
>>> mv[2:3]
