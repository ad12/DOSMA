.. _basic_usage:

.. py:currentmodule:: dosma

**This guide is still under construction**

Basic Usage
-----------

Dosma is designed for simple imaging I/O, registration, quantitative fitting, and AI-based image processing. 

Dosma does not bundle Tensorflow and Keras installation by default.
To enable  support, you must install these libraries as an additional step.

We use the following abbreviations for libraries:

>>> import numpy as np
>>> import dosma as dm


Image I/O
=========================

Dosma provides data readers and writers to allow you to read/write image data stored in NIfTI and DICOM standards.
These I/O tools create or write from the Dosma image class :class:`MedicalVolume`.

For example to load a DICOM image series, which has multiple echos, with each echo corresponding to a volume,
we can do:

>>> from dosma import DicomReader:
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

>>> from dosma import NiftiReader
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


NumPy Interoperability
========================================

In addition to standard shape-preserving universal functions (ufuncs) described above,
:class:`MedicalVolume` also support a subset of other numpy functions that, like the ufuncs,
operate on the pixel data in the medical volume:

- Boolean Functions: :func:`numpy.all`, :func:`numpy.any`, :func:`numpy.where`
- Statistics functions: :func:`numpy.mean`, :func:`numpy.sum`, :func:`numpy.std`, :func:`numpy.amin`, :func:`numpy.amax`, :func:`numpy.argmax`, :func:`numpy.argmin`
- Rounding functions: :func:`numpy.round`, :func:`numpy.around`, :func:`numpy.round_`
- NaN functions: :func:`numpy.nanmean`, :func:`numpy.nansum`, :func:`numpy.nanstd`, :func:`numpy.nan_to_num`

For example, ``np.all(mv)`` is equivalent to ``np.all(mv.volume)``. Note, headers are not deep copied.
NumPy operations that reduce spatial dimensions are not supported. For example, a 3D volume ``mv`` cannot
be summed over any two of the first three axes:

>>> np.sum(mv, 0)  # this will raise an error
>>> np.sum(mv)  # this will return a scalar


(BETA) Choosing A Computing Device
========================================

Dosma provides a device class :class:`dosma.Device`, which allows you to specify which device
to use for :class:`MedicalVolume` operations. It extends the Device class from `CuPy <https://cupy.dev/>`_.
To enable GPU computing support, install the correct build for CuPy on your machine.

To move a MedicalVolume to GPU 1, you can use the :meth:`MedicalVolume.to` method:

>>> mv_gpu = mv.to(dm.Device(1))

You can also move the image back to the cpu:

>>> mv_cpu = mv_gpu.to(dm.Device(-1))  # or mv_gpu.cpu()

If the device is already on the specified device, the same object is returned.
Note, some functionality such as curve fitting (:class:`dosma.curve_fit`), image registration,
and image I/O are not supported with images on the GPU.
