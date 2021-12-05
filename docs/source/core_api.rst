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

   dosma.core.numpy_routines.all_np
   dosma.core.numpy_routines.amax
   dosma.core.numpy_routines.amin
   dosma.core.numpy_routines.any_np
   dosma.core.numpy_routines.argmax
   dosma.core.numpy_routines.argmin
   dosma.core.numpy_routines.around
   dosma.core.numpy_routines.clip
   dosma.core.numpy_routines.concatenate
   dosma.core.numpy_routines.expand_dims
   dosma.core.numpy_routines.may_share_memory
   dosma.core.numpy_routines.mean_np
   dosma.core.numpy_routines.nan_to_num
   dosma.core.numpy_routines.nanargmax
   dosma.core.numpy_routines.nanargmin
   dosma.core.numpy_routines.nanmax
   dosma.core.numpy_routines.nanmean
   dosma.core.numpy_routines.nanmin
   dosma.core.numpy_routines.nanstd
   dosma.core.numpy_routines.nansum
   dosma.core.numpy_routines.ones_like
   dosma.core.numpy_routines.pad
   dosma.core.numpy_routines.shares_memory
   dosma.core.numpy_routines.squeeze
   dosma.core.numpy_routines.stack
   dosma.core.numpy_routines.std
   dosma.core.numpy_routines.sum_np
   dosma.core.numpy_routines.where
   dosma.core.numpy_routines.zeros_like

Standard universal functions that act element-wise on the array are also supported.
A (incomplete) list is shown below:

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - numpy.power
     - numpy.sign
     - numpy.remainder
     - numpy.mod
     - numpy.abs
   * - numpy.log
     - numpy.exp
     - numpy.sqrt
     - numpy.square
     - numpy.reciprocal
   * - numpy.sin
     - numpy.cos
     - numpy.tan
     - numpy.bitwise_and
     - numpy.bitwise_or
   * - numpy.isfinite
     - numpy.isinf
     - numpy.isnan
     - numpy.floor
     - numpy.ceil


Image I/O
---------------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.read
   dosma.write
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
