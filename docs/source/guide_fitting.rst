.. _guide_fitting:

**This guide is still under construction**

Fitting
-----------

Dosma supports cpu-parallelizable quantitative fitting based on
`scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_. 

To perform generic fitting to any array-like object using an arbitrary model function, we can use
:func:`dosma.curve_fit`. For example, we can use this function to fit an array to a
monoexponential model ``y = a * exp(b * x)`` using a maximum of 4 workers:

>>> from dosma import curve_fit, monoexponential
>>> curve_fit(monoexponential, x, y, num_workers=4)

Quantitative fitting is quite common in medical image analysis. For example,
quantitative MRI (qMRI) has enabled computing voxel-wise relaxation parameter maps
(e.g. |T2|, |T1rho|, etc.). We can fit a monoexponential model for each voxel across these registered_images,
where ``tc0`` is the initial guess for parameter :math:`-\frac{1}{b}` in the monoexponential model:

>>> from dosma import MonoExponentialFit
>>> tc0 = 30.0
>>> echo_times = np.asarray([10.0, 20.0, 50.0])
>>> fitter = MonoExponentialFit(tc0=tc0, num_workers=4)
>>> tc, r2_map = fitter.fit(echo_times, images)

If you don't have a good initial guess for ``tc0`` or expect the initial guess to be dependent on the voxel being fit
(which is often the case), you can specify that the initial guess should be determined based on results from a
polynomial fit over the log-linearized form of the monoexponential equation ``log(y) = log(a) - x/tc``:

>>> from dosma import MonoExponentialFit
>>> tc0 = "polyfit"
>>> echo_times = np.asarray([10.0, 20.0, 50.0])
>>> fitter = MonoExponentialFit(tc0=tc0, num_workers=4)
>>> tc, r2_map = fitter.fit(echo_times, images)

Custom model functions can also be provided and used with :class:`dosma.curve_fit` and :class:`dosma.CurveFitter` (recommended),
a class wrapper around :class:`dosma.curve_fit` that handles :class:`MedicalVolume` data and supports additional post-processing
on the fitted parameters. The commands below using :class:`dosma.CurveFitter` and :class:`dosma.curve_fit` are equivalent to the
``fitter`` above:

>>> from dosma import CurveFitter
>>> cfitter = CurveFitter(
... monoexponential, p0=(1.0, -1/tc0), num_workers=4, nan_to_num=0,
... out_ufuncs=[None, lambda x: -1/x], out_bounds=(0, 100))
>>> popt, r2_map = cfitter.fit(echo_times, images)
>>> tc = popt[..., 1]

>>> from dosma import curve_fit
>>> curve_fit(monoexponential, echo_times, [x.volume for x in images], p0=(1.0, -1/tc0), num_workers=4)

Non-linear curve fitting often requires carefully selected parameter initialization. In cases where
non-linear curve fitting fails, polynomial fitting may be more effective. Polynomials can be fit to
the data using :func:`dosma.polyfit` or :class:`dosma.PolyFitter` (recommended),
which is the polynomial fitting equivalent of ``CurveFitter``. Because polynomial fitting can also be
done as a single least squares problem, it may also often be faster than standard curve fitting.
The commands below use ``PolyFitter`` to fit to the log-linearized monoexponential fit
(i.e. ``log(y) = log(a) + b*x`` to some image data:

>>> from dosma import PolyFitter
>>> echo_times = np.asarray([10.0, 20.0, 50.0])
>>> pfitter = PolyFitter(deg=1, nan_to_num=0, out_ufuncs=[None, lambda x: -1/x], out_bounds=(0, 100))
>>> log_images = [np.log(img) for img in images]
>>> popt, r2_map = pfitter.fit(echo_times, log_images)
>>> tc = popt[..., 0]  # note ordering of parameters - see numpy.polyfit for more details.

We can also use the polyfit estimates to initialize the non-linear curve fit. For monoexponential
fitting, we can do the following:

>>> from dosma import CurveFitter, PolyFitter
>>> echo_times = np.asarray([10.0, 20.0, 50.0])
>>> pfitter = PolyFitter(deg=1, r2_threshold=0, num_workers=0)
>>> log_images = [np.log(img) for img in images]
>>> popt_pf, _ = pfitter.fit(echo_times, log_images)
>>> cfitter = CurveFitter(monoexponential, r2_threshold=0.9, nan_to_num=0, out_ufuncs=[None, lambda x: -1/x], out_bounds=(0, 100))
>>> popt, r2 = cfitter.fit(echo_times, images, p0={"a": popt_pf[..., 1], "b": popt_pf[..., 0]})
>>> tc = popt[..., 1] 

.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`
