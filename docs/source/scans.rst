.. _scans:

Scans (dosma.scan_sequences)
================================================================================
DOSMA currently supports 4 MRI scan types:

1. Quantitative DESS (qDESS)
2. CubeQuant
3. MAPSS
4. UTE Cones

Each scan implements a subset of the following actions:

1. **Segment** tissues from scan
2. **Interregister** to register between lower resolution (moving) and higher resolution (target) scans
3. **Quantitative fitting** for voxel-wise parameter maps.

.. automodule::
   dosma.scan_sequences

.. autosummary::
   :toctree: generated

   dosma.scan_sequences.ScanSequence
   dosma.scan_sequences.QDess
   dosma.scan_sequences.CubeQuant
   dosma.scan_sequences.Mapss
   dosma.scan_sequences.Cones

Below we briefly discuss the different scan types and associated actions.

qDESS
--------------------------------------------------------------------------------
Quantitative double echo in steady state (qDESS) is a high-resolution scan that has shown high efficacy for analytic
|T2| mapping :cite:`sveinsson2017simple`. Because of its high resolution, qDESS scans have been shown to be good candidates for automatic
segmentation.

DOSMA supports both automatic segmentation and analytical |T2| solving for qDESS scans. Automated segmentation uses
pre-trained convolutional neural networks (CNNs).


CubeQuant (3D FSE)
--------------------------------------------------------------------------------
Cubequant is a 3D fast-spin-echo (FSE) |T1rho|-weighted sequence. Acquisitions between spin-locks are
susceptible to motion, and as a result, volumes within the scan have to be registered to each other
(i.e. *intra*-registered).

Moreover, CubeQuant scans often have lower resolution to increase SNR in practice. Because of the
low-resolution, these scans are often registered to higher resolution target scans :cite:`jordan2014variability`.

By default, DOSMA intraregisters volumes acquired at different spin-locks to one another. This framework also supports
both registration between scan types (interregistration) and |T1rho| fitting.

Because registration is sensitive to the target scan type, different registration approaches may work better with
different scan types. By default, the registration approaches are optimized to register CubeQuant scans to qDESS scans.


3D MAPSS (SPGR)
--------------------------------------------------------------------------------
Magnetization‐prepared angle‐modulated partitioned k‐space spoiled gradient echo snapshots (3D MAPSS) is a spoiled
gradient (SPGR) sequence that reduce specific absorption rate (SAR), increase SNR, and reduce the extent of
retrospective correction of contaminating |T1| effects :cite:`li2008vivo`.

The MAPSS sequence can be used to estimate both |T1rho| and |T2| quantitative values. Like CubeQuant scans, MAPSS scans
must also be intraregistered to ensure alignment between all volumes acquired at different echos and spin-lock times.

DOSMA automatically performs intraregistration among volumes within the MAPSS scan. |T2| and |T1rho| fitting is also
supported.


UTE Cones
--------------------------------------------------------------------------------
Ultra-short echo time (UTE) Cones (or Cones) is a |T2star|-weighted sequence. In practice, many of these scans are low
resolution.

DOSMA supports interregistration between Cones and other scan sequences; however, registration files are optimized for
registration to qDESS. |T2star| fitting is also supported.


.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`