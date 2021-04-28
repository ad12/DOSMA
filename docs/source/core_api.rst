.. _core_api:

Core API (dosma.core)
================================================================================

MedicalVolume
---------------------------
.. _core_api_medicalvolume:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.MedicalVolume


Numpy Routines
---------------------------
.. _core_api_numpy_routines:

Numpy operations that are supported on MedicalVolumes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.core.numpy_routines


Image I/O
---------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.NiftiReader
   dosma.NiftiWriter
   dosma.DicomReader
   dosma.DicomWriter


Image Orientation
---------------------------
.. automodule::
   dosma.core.orientation

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.core.orientation.to_affine
   dosma.core.orientation.get_transpose_inds
   dosma.core.orientation.get_flip_inds
   dosma.core.orientation.orientation_nib_to_standard
   dosma.core.orientation.orientation_standard_to_nib


Image Registration
---------------------------
For details on using registration, see the :ref:`Registration Guide <guide_registration>`.

.. automodule::
   dosma.core.registration

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.register
   dosma.apply_warp
   dosma.symlink_elastix
   dosma.unlink_elastix


Fitting
---------------------------
For details on using fitting functions, see the :ref:`Fitting Guide <guide_fitting>`.

.. automodule::
   dosma.core.fitting

General fitting functions:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.curve_fit
   dosma.polyfit
   dosma.core.fitting.monoexponential
   dosma.core.fitting.biexponential

Fitter classes:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.CurveFitter
   dosma.PolyFitter
   dosma.MonoExponentialFit


Device
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.Device
   dosma.get_device
   dosma.to_device


Preferences
-------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.defaults._Preferences
   dosma.preferences


(BETA) Quantitative Values
---------------------------
Utilities for different quantitative parameters.
Note, this feature is in beta and will likely change in future releases.

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.core.quant_vals.QuantitativeValue
   dosma.core.quant_vals.T1Rho
   dosma.core.quant_vals.T2
   dosma.core.quant_vals.T2Star
