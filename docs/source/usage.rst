.. _usage_cli:

Application
================================================================================
This page discusses general usage tips for both the command line and graphical user interface (GUI).
Because the GUI is a wrapper around the command line and is typically easier to use, we will focus on detailing how to
use DOSMA from the command line.

The details here assume that :ref:`installation <installation>` and :ref:`verification <install-verification>`
worked as expected.

.. _usage-session:

Starting A Session
--------------------------------------------------------------------------------
To being a session using DOSMA from the command-line or GUI, follow the step below:

1. Open new Terminal window.
2. Activate DOSMA Anaconda environment::

    $ conda activate dosma_env

3. Run DOSMA::

    # Run DOSMA from command line.
    $ python -m dosma.cli <ARGS...>

    # Or run DOSMA as a user interface (UI).
    $ python -m dosma.app

.. note::

    If you are using a remote connection, enable X11 port-forwarding to execute Step 4. If it is not enabled, the GUI
    cannot be used for remote connections.


.. _usage-cli:

Command-Line (Terminal)
--------------------------------------------------------------------------------
DOSMA was designed to be a script-friendly library, where wrapper scripts can be used to
execute and parallelize DOSMA computations. As scripting is often done using command-line enabled scripts (bash, shell,
perl, etc.), DOSMA has a command-line interface.

The DOSMA command-line parser hierarchy funnels from scan type to action. Scan types are specified first and are
followed by the action to execute and corresponding arguments. For example, the code below computes |T2| for femoral
cartilage (fc) and patellar cartilage (pc) from qDESS scans::

    $ python -m dosma.cli --dicom subject01/dicoms/qdess --save subject01/data qdess --fc --pc generate_t2_map

Scans
^^^^^
:ref:`Scans <scans>` and corresponding actions are specified as subparsers. For example, the code below is the skeleton
for segmenting tissues in qDESS scans::

    $ python -m dosma.cli qdess segment ....


Tissues
^^^^^^^
:ref:`Tissues <tissues>` are specified as arguments of the subparser using the tissue's abbreviated name.
For example, ``... qdess --fc ...`` would be performing a qDESS-specific action on femoral cartilage.

To analyze multiple tissues, add additional flags. For example, the argument below
generates a |T2| map for both femoral and patellar cartilage::

    $ python -m dosma.cli --dicom subject01/dicoms/qdess --save subject01/data qdess --fc --pc generate_t2_map

Analysis
^^^^^^^^
DOSMA also has a command-line subparser for creating visualizations and analyze quantitative values for tissues.
This subparser is identified by the anatomical region being analyzed.

DOSMA currently supports knee-related analyses. To run the analysis on |T2| for femoral cartilage and tibial cartilage, use the following skeleton::

    $ python -m dosma.cli --load subject01/data knee --fc --tc --t2

Help
^^^^
To get more information, start a new session and run ``python -m dosma.cli --help``. To get the help menu of subparsers, add the
subparser name to the help menu::

    # Print high-level help menu for DOSMA.
    $ python -m dosma.cli --help

    # Print help menu for qDESS scan.
    $ python -m dosma.cli qdess --help

    # Print help menu for qDESS segmentation.
    $ python -m dosma.cli qdess segment --help

Examples
^^^^^^^^
We detail some examples that could be useful for analyzing data. Note there any any number of combinations with how the
data is analyzed. Below are just examples for how they have commonly been used by current users.

We assume the folder structure looks something like below:

::

    research_data
        ├── patient01
            ├── qdess (qDESS dicoms)
            |    └── I0001.dcm
            |    └── I0002.dcm
            |    └── I0003.dcm
            |    ....
            ├── cubequant (CubeQuant dicoms)
            ├── cones (UTE Cones dicoms)
        ├── patient02
            ├── mapss (MAPSS dicoms)
        ├── patient03
        ├── weights (segmentation weights)
            ├── oai_unet2d
            ├── iwoai-2019-t6
            ├── iwoai-2019-t6-normalized



qDESS
#####
Analyze patient01's femoral cartilage |T2| properties using qDESS sequence*::

    # 1. Calculate 3D T2 map - suppress fat and fluid to reduce noise
    $ python -m dosma.cli --dicom research_data/patient01/dess --save research_data/patient01/data qdess --fc t2 --suppress_fat --suppress_fluid

    # 2. Segment femoral cartilage on root sum of squares (RSS) of two echo qDESS echos using OAI 2D U-Net model.
    $ python -m dosma.cli --dicom research_data/patient01/dess --save research_data/patient01/data qdess --fc segment --rss --weights_dir weights/oai_unet2d --model oai-unet2d

    # 3. Calculate/visualize T2 for femoral cartilage
    $ python -m dosma.cli --load research_data/patient01/data --save research_data/patient01/data knee --fc --t2


CubeQuant
#########
Analyze patient01 femoral cartilage |T1rho| properties using Cubequant sequence::

    # 1. Register cubequant volumes (i.e. different spin-lock-times) to one another (intraregistration)
    $ python -m dosma.cli --dicom research_data/patient01/cubequant --save research_data/patient01/data cubequant intraregister

    # 2. Register cubequant volume to first echo of qDESS sequence
    $ python -m dosma.cli --load research_data/patient01/data cubequant --fc interregister --target_path research_data/patient01/data/dess/echo1.nii.gz --target_mask research_data/patient01/data/fc/fc.nii.gz

    # 3. Calculate 3D T1-rho map only for femoral cartilage region
    $ python -m dosma.cli --load research_data/patient01/data cubequant --fc t1_rho  --mask_path research_data/patient01/data/fc/fc.nii.gz

    # 4. Calculate/visualize T1-rho for femoral cartilage
    $ python -m dosma.cli --load research_data/patient01/data --fc --t1_rho


UTE Cones
#########
Analyze patient01 femoral cartilage |T2star| properties using UTE Cones sequence::

    # 1. Register cones volume to first echo of qDESS sequence
    $ python -m dosma.cli --dicom research_data/patient01/cones --save research_data/patient01/data cones --fc interregister --target_path research_data/patient01/data/dess/echo1.nii.gz --target_mask research_data/patient01/data/fc/fc.nii.gz

    # 2. Calculate 3D T2-star map only for femoral cartilage region
    $ python -m dosma.cli --load research_data/patient01/data cones --fc t2_star --mask_path research_data/patient01/data/fc/fc.nii.gz

    # 3. Calculate/visualize T1-rho for femoral cartilage
    $ python -m dosma.cli --load research_data/patient01/data knee --fc --t2_star


MAPSS
#####
Analyze patient02 femoral cartilage |T1rho| and |T2| properties using MAPSS sequence::

    # 1. Fit T1-rho for whole volume
    $ python -m dosma.cli --dicom research_data/patient02/mapss --save research_data/patient02/data mapss --fc t1_rho

    # 2. Fit T2 for whole volume
    $ python -m dosma.cli --dicom research_data/patient02/mapss --save research_data/patient02/data mapss --fc t2

    # 3. Manually segment femoral cartilage and store in appropriate folders.

    # 4. Calculate/visualize T1-rho and T2 for femoral cartilage
    $ python -m dosma.cli --load research_data/patient01/data knee --fc --t2_star


.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`
