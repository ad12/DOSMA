.. _installation:

Installation
================================================================================

This page provides a step-by-step overview of creating a virtual environment,
installing DOSMA via `pip`, and verifying the install.


Anaconda
--------------------------------------------------------------------------------
Please install the `Anaconda <https://www.anaconda.com/download/>`_ virtual environment manager.


.. _install-setup:

Setup
--------------------------------------------------------------------------------
The following steps will create an Anaconda environment (``dosma_env``).

1. Open a Terminal/Shell window
2. Create the `dosma_env` environment::

    $ conda create -n dosma_env python=3.7

3. Install dosma via pip::

    $ pip install dosma

4. Complete the `DOSMA questionnaire <https://forms.gle/sprthTC2swyt8dDb6>`_.

If you want to update your dosma version, run ``pip install --upgrade dosma``.


Segmentation
############
DOSMA currently supports automatic deep learning segmentation methods. These methods use pre-trained weights for
segmenting tissues in specific scans. Currently, segmentation for quantitative double echo in steady state (qDESS) scans
is supported for knee articular cartilage and meniscus.

If you will be using this functionality, please follow the instructions below.

1. Request access using this `Google form <https://goo.gl/forms/JlxgS3aoUeeUUlVh2>`_
   *and* email arjundd (at) <standard Stanford email domain>

2. Save these weights in an accessible location. **Do not rename these files**.

We understand this process may be involved and are actively working on more effective methods to distribute these
weights.

.. _install-setup-registration:

Registration
############
Registration between scans in DOSMA is supported through Elastix and Transformix. If you plan on using the registration,
follow the instructions below:

1. Download `elastix <https://elastix.lumc.nl/download.php>`_
2. Follow instructions on adding elastix/transformix to your system path

On Ubuntu 18.04 Elastix version 5.0.1 does not work properly. Elastix 4.9.0 is recommended.

If you are using a MacOS system, you may run into path issues with elastix (see
`this discussion <https://github.com/almarklein/pyelastix/issues/9>`_). To fix
this, we can use the `dosma.symlink_elastix` to create
appropriate symbolic links to files causing issues:

    $ conda activate dosma_env; python
    >>> from dosma import symlink_elastix
    >>> symlink_elastix()

Note you will need to run this every time you update elastix/transformix paths
on your machine.

.. _install-verification:

Verification
--------------------------------------------------------------------------------
1. Open new Terminal window.
2. Activate DOSMA Anaconda environment::

    $ conda activate dosma_env

3. Run DOSMA from the command-line (cli). You should see a help menu output::

    $ python -m dosma.cli --help

4. Run DOSMA as an UI application (app). You should see a UI window pop-up::

    $ python -m dosma.app


Updating DOSMA
--------------------------------------------------------------------------------
If you have used an earlier stand-alone of DOSMA (v0.0.11 or earlier), you may
already have a ``dosma_env`` virtual environment. Please delete this environment
and reinstall follows steps in setup :ref:`Setup <install-setup>`.

For those (v0.0.12 or later) having installed dosma via ``pip``, you can update
dosma using::

    $ pip install --upgrade dosma
