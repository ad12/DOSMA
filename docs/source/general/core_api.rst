.. _documentation:

Core API
================================================================================

MedicalVolume
---------------------------
.. autosummary::
   :toctree:
   :nosignatures:

   dosma.data_io.MedicalVolume


Image I/O
---------------------------
.. autosummary::
   :toctree:
   :nosignatures:

   dosma.data_io.NiftiReader
   dosma.data_io.NiftiWriter
   dosma.data_io.DicomReader
   dosma.data_io.DicomWriter


Image Orientation
---------------------------
.. automodule::
   dosma.data_io.orientation

.. autosummary::
   :toctree:
   :nosignatures:

   dosma.data_io.orientation.get_transpose_inds
   dosma.data_io.orientation.get_flip_inds
   dosma.data_io.orientation.orientation_nib_to_standard
   dosma.data_io.orientation.orientation_standard_to_nib


Image Registration
---------------------------
For details on using registration, see the :ref:`Registration Guide <guide_registration>`.

.. automodule::
   dosma.utils.registration

.. autosummary::
   :toctree:
   :nosignatures:

   dosma.utils.registration.register
   dosma.utils.registration.apply_warp
   dosma.utils.registration.symlink_elastix
   dosma.utils.registration.unlink_elastix


Fitting
---------------------------
For details on using fitting functions, see the :ref:`Fitting Guide <guide_fitting>`.

.. automodule::
   dosma.utils.fits

General fitting functions:

.. autosummary::
   :toctree:
   :nosignatures:

   dosma.utils.fits.curve_fit
   dosma.utils.fits.monoexponential
   dosma.utils.fits.biexponential

Fitter classes:

.. autosummary::
   :toctree:
   :nosignatures:

   dosma.utils.fits.CurveFitter
   dosma.utils.fits.MonoExponentialFit

