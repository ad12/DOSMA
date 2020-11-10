.. _installation:

Installation
================================================================================

This page provides an overview of installing *DOSMA* as a stand-alone application
and as a local Python package using pip.

.. note::

   The commands in this documentation are for Linux or Mac OS. If you work on Windows
   use the `Linux bash <https://itsfoss.com/install-bash-on-windows/>`_ provided in Windows 10.


Anaconda
--------------------------------------------------------------------------------
Please install the `Anaconda <https://www.anaconda.com/download/>`_ virtual environment manager.


.. _install-setup:

Setup
--------------------------------------------------------------------------------
The following steps will create an Anaconda environment and a shortcut for running DOSMA from the command-line.
**Avoid spaces in file paths.**

1. Download the `latest release <https://github.com/ad12/DOSMA/releases>`_ to a non-temporary location (i.e. not the `Downloads` folder)
2. Open the project folder in the Terminal
3. Navigate to ``bin`` folder::

    $ cd bin

4. Initialize ``setup`` executable::

    # Initialize executable.
    $ chmod +x setup

5. Run ``setup``::

    # Run executable.
    $ ./setup

6. Close terminal window.
7. Complete `DOSMA questionnaire <https://forms.gle/sprthTC2swyt8dDb6>`_.

If you want to update your Anaconda environment, run ``./setup -f`` in step 4.

pip install
###########
The library can now also be installed via ``pip``, although only as a local library.
We are in the process of hosting DOSMA on PyPi.

To install as a library, navigate to the project folder in the Terminal and run the commands below::

    # Activate your environment
    $ conda activate dosma_env

    # pip install in editable format
    $ python -m pip install -e ./

Segmentation
############
DOSMA currently supports automatic deep learning segmentation methods. These methods use pre-trained weights for
segmenting tissues in specific scans. Currently, segmentation for quantitative double echo in steady state (qDESS) scans
is supported for knee articular cartilage and meniscus.

If you will be using this functionality, please follow the instructions below.

1. Request access using this `Google form <https://goo.gl/forms/JlxgS3aoUeeUUlVh2>`
   and email arjundd (at) <standard Stanford email domain>

2. Save these weights in an accessible location. **Do not rename these files**.

We understand this process may be involved and are actively working on more effective methods to distribute these
weights.


Registration
############
Registration between scans in DOSMA is supported through Elastix and Transformix. If you plan on using the registration,
follow the instructions below:

1. Download `elastix <http://elastix.isi.uu.nl/download.php>`_
2. Follow instructions on adding elastix/transformix to your system path
3. Copy (not move) the file ``libANNlib.dylib`` to the DOSMA project folder downloaded earlier.

.. _install-verification:

Verification
--------------------------------------------------------------------------------
1. Open new Terminal window.
2. Activate DOSMA Anaconda environment::

    $ conda activate dosma_env

3. Run DOSMA from the command-line. You should see a help menu output::

    $ dosma --help

4. Run DOSMA as an UI application. You should see a UI window pop-up::

    $ dosma --app


Updating DOSMA
--------------------------------------------------------------------------------
To update DOSMA to the latest version, delete the ``DOSMA`` folder and follow the
instructions in :ref:`Setup <install-setup>`.

To use a specific DOSMA version, download the source code for the desired version
and follow the setup instructions.
