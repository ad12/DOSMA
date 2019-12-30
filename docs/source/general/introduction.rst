.. _introduction:

Introduction
================================================================================
DOSMA is an open-source Python library and application for musculoskeletal (MSK) MRI analysis.

In offering support for multiple scan types and tissues, DOSMA is built to streamline rapid multi-scan analysis of MSK
anatomy. Because DOSMA is abstracted to be a framework, it can be used to write analysis protocols that can be run for
different combination of scans, reducing overhead of computation.

For example, the analysis workflow for a combination
of quantitative DESS, CubeQuant (3D fast spin echo), and ultra-short echo time Cones scans for multiple patients
(shown below) can be done in 7 lines of code:

.. figure:: ../figures/workflow.png
   :align: center
   :alt: Example workflow for analyzing multiple scans per patient
   :figclass: align-center

   Example workflow for analyzing 1. quantitative DESS (qDESS), a |T2|-weighted sequence,
   2. CubeQuant, a |T1rho|-weighted sequence, and ultra-short echo time Cones, a |T2star|
   weighted sequence.

Workflow
--------------------------------------------------------------------------------
DOSMA uses various modules to handle MSK analysis for multiple scan types and tissues:

- **Scan** modules declare scan-specific actions (fitting, segmentation, registration, etc).
- **Tissue** modules handle visualization and analysis optimized for different tissues.
- **Analysis** modules abstract different methods for performing different actions (different segmentation methods, fitting methods, etc.)

Features
--------------------------------------------------------------------------------

Dynamic Input/Output (I/O)
^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading and writing medical images relies on standardized data formats.
The Digital Imaging and Communications in Medicine (DICOM) format has been the international
standard for medical image I/O. However, header information is memory intensive and
and may not be useful in cases where only volume information is desired.

The Neuroimaging Informatics Technology Initiative (NIfTI) format is useful in these cases.
It stores only volume-specific header information (rotation, position, resolution, etc.) with
the volume.

DOSMA supports the use of both formats. However, because NIfTI headers do not contain relevant scan
information, it is not possible to perform quantitative analysis that require this information.
Therefore, we recommend using DICOM inputs, which is the standard output of acquisition systems,
when starting processing with DOSMA.

By default,  volumes (segmentations, quantitative maps, etc.) are written in the NIfTI format.
The default output file format can be changed in the :ref:`preferences <faq-citation>`.

Multiple Orientations
^^^^^^^^^^^^^^^^^^^^^
We support analyzing volumes acquired in any plane and support automatic
reformatting to the expected plane during computation.

Our machine learning methods are trained using sagittal acquired images,
so performance may vary for images acquired in different planes
(caused by differences in in-plane resolution, FOV, etc.).


Disclaimers
--------------------------------------------------------------------------------

Using Deep Learning
^^^^^^^^^^^^^^^^^^^
All weights/parameters trained for any task are likely to be most closely correlated to data used for training.
If scans from a particular sequence were used for training, the performance of those weights are likely optimized
for that specific scan prescription (resolution, TR/TE, etc.). As a result, they may not perform as well on segmenting images
acquired using different scan types.

If you do train weights for any deep learning task that you would want to include as part of this repo, please provide
a link to those weights and detail the scanning parameters/sequence used to acquire those images.

OS Compatibility
^^^^^^^^^^^^^^^^
This library has been optimized for use on Mac/Linux operating systems. We are working on a Windows-specific
solution for the library (`Issue #39 <https://github.com/ad12/DOSMA/issues/39>`_). In the meantime, the
`Linux bash <https://itsfoss.com/install-bash-on-windows/>`_ can be used.

.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`

