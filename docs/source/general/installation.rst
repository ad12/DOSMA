.. _installation:

Installation
================================================================================


.. raw:: html

  This page provides an overview of installing <i>DOSMA</i> as a stand-alone application.


.. note::

   The commands in this documentation are for Linux or Mac OS. If you work on Windows:

   - Substitute ``/`` with ``\``
   - Use the python :ref:`terminal <terminal>` provided in the Anaconda distribution



Anaconda
--------------------------------------------------------------------------------
Please install the `Anaconda <https://www.anaconda.com/download/>`_ virtual environment manager.


Setup
--------------------------------------------------------------------------------
The following steps will create an Anaconda environment and a shortcut for running DOSMA from the command-line.
**Avoid spaces in file paths.**

1. Download the DOSMA `repository <https://github.com/ad12/DOSMA>`_ to a non-temporary location (i.e. not the `Downloads` folder)
2. Open the DOSMA directory in the Terminal
3. Navigate to ``bin`` folder::

    $ cd bin

4. Initialize and run ``setup`` executable::

    # Initialize executable.
    $ chmod +x setup

    # Run executable. Use '-f' flag to update Anaconda environment.
    $ ./setup

5. Close terminal window.
6. Open browser and complete `DOSMA questionnaire <https://forms.gle/sprthTC2swyt8dDb6>`_.

If you want to update your Anaconda environment, run ``./setup -f`` in step 4.


Segmentation
############
DOSMA currently supports automatic deep learning segmentation methods. These methods use pre-trained weights for
segmenting tissues in specific scans. Currently, segmentation for quantitative double echo in steady state (qDESS) scans
is supported for knee articular cartilage and meniscus.

If you will be using this functionality, please follow the instructions below.

1. Request access using this `Google form <https://goo.gl/forms/JlxgS3aoUeeUUlVh2>`_.
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

.. note::

    If you are using a remote connection, enable X11 port-forwarding to execute Step 4. If it is not enabled, the GUI
    cannot be used for remote connections.