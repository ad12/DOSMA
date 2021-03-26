.. _guide_fitting:

**This guide is still under construction**

Fitting
-----------

Dosma supports cpu-parallelizable quantitative fitting based on
`scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_. 

To perform generic fitting to any array-like object using an arbitrary model function, we can use
:func:`dosma.utils.fits.curve_fit`. For example, we can use this function to fit an array to a
monoexponential model ``y = a * exp(b * x)`` using a maximum of 4 workers:

>>> from dosma.utils.fits import curve_fit, monoexponential
>>> curve_fit(monoexponential, x, y, num_workers=4)

Quantitative fitting is quite common in medical image analysis. For example,
quantitative MRI (qMRI) has enabled computing voxel-wise relaxation parameter maps
(e.g. |T2|, |T1rho|, etc.). We can fit a monoexponential model for each voxel across these registered_images,
where ``tc0`` is the initial guess for parameter :math:`-\frac{1}{b}` in the monoexponential model:

>>> from dosma.utils.fits import MonoExponentialFit
>>> tc0 = 30.0
>>> echo_times = np.asarray([10.0, 20.0, 50.0])
>>> fitter = MonoExponentialFit(echo_times, images, tc0=tc0, num_workers=4)
>>> quant_map, r2_map = fitter.fit()

Custom model functions can also be provided and used with ``curve_fit`` and ``CurveFitter`` (recommended),
a class wrapper around ``curve_fit`` that handles ``MedicalVolume`` data and supports additional post-processing
on the fitted parameters. The commands below using ``CurveFitter`` and ``curve_fit`` are equivalent to the
``fitter`` above:

>>> from dosma.utils.fits import CurveFitter
>>> cfitter = CurveFitter(
... monoexponential, p0=(1.0, -1/tc0), num_workers=4, nan_to_num=0,
... out_ufuncs=[None, lambda x: -1/x], out_bounds=(0, 100))
>>> popt, r2_map = cfitter.fit(echo_times, images)
>>> quant_map = popt[..., 1]

>>> from dosma.utils.fits import curve_fit
>>> curve_fit(monoexponential, echo_times, [x.volume for x in images], p0=(1.0, -1/tc0), num_workers=4)

.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`
