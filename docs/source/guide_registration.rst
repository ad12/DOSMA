.. _guide_registration:

**This guide is still under construction**

Image Registration
------------------

Dosma supports image registration using the Elastix and Transformix by creating a
wrapper around the standard command-line usage. In addition to multi-threading, Dosma
supports true parallel execution when registering multiple volumes to a target.

Elastix/Transformix must be installed and configured on your machine. See
:ref:`the setup guide <install-setup-registration>` for more information

To register moving image(s) to a target image, we can use :class:`dosma.register`:

>>> from dosma import register
>>> out = register(target, moving, "/path/to/elastix/file", "/path/to/save", return_volumes=True)
>>> registered_images = out["volumes"]

To use multiple workers, we can pass the ``num_workers`` argument. Note that ``num_workers``
parallelizes registration when there are multiple moving images. The true number of parallel
processes are equivalent to ``min(num_workers, len(moving))``. To increase the number of threads
used per, use ``num_threads``.

To transform moving image(s) using a transformation file, we can use :class:`dosma.apply_warp`:

>>> from dosma import apply_warp
>>> transformed_image = apply_warp(image, transform="/path/to/transformation/file")

Often we may want to copy the final transformation file produced during registration to transform
other volumes:

>>> reg_out = register(target, moving, "/path/to/elastix/file", "/path/to/save", return_volumes=True)
>>> warp_out = apply_warp(other_moving, transform=out_reg["outputs"].transform)
