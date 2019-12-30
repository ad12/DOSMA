.. _cli:

Command Line Documentation
================================================================================
This page is a more exhaustive coverage of running DOSMA from the command line.

Help menus for different scans and actions are shown. In the code blocks, code under the ``In`` header is what was
typed into the terminal window. Text under the ``Out`` header is the help menu output for that specific module.

To run the code, :ref:`start a new session <usage-session>`.

Basic Menu
--------------------------------------------------------------------------------
When using DOSMA to for the first time on a new series of DICOMs, specify the path to the series
folder using the ``--dicom`` flag. Also specify the directory to save DOSMA outputs using ``--save``::

    # Path to series 007 in subject01 folder.
    $ dosma --dicom subject01/007 --save subject01/data

Once, data has been processed for a particular series, you can specify the load path (``--load``)
to be the save path. If loading and storing the data in the same directory, the save directory (``--save``) doesn't
need to be specified::

    # Loading data for subject01, series 007. Data will be saved in the directory specified by --load.
    $ dosma --load subject01/007 ...

In practice, either ``--dicom`` or ``--load`` should be specified, not both.

If doing computationally expensive operations that ar GPU-compatible (automatic segmentation), use the ``--gpu`` flag to
specify the GPUs to use. The code below uses GPUs ``0`` and ``1``::

    $ dosma --gpu "0,1"

The general help menu is shown below:
.. code-block:: bash

    In:
    ---------
    dosma --help

    Out:
    ---------
    usage: DOSMA [-h] [--debug] [--d [D]] [--l [L]] [--s [S]] [--ignore_ext]
                 [--split_by [G]] [--gpu [G]] [--df [{dicom,nifti}]] [--r2 [T]]
                 [--dpi [DPI]]
                 [--vf [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}]]
                 {qdess,cubequant,mapss,knee} ...

    A deep-learning powered open source MRI analysis pipeline

    positional arguments:
      {qdess,cubequant,mapss,knee}
                            sub-command help
        qdess               analyze qdess sequence
        cubequant           analyze cubequant sequence
        mapss               analyze mapss sequence
        knee                calculate/analyze quantitative data for knee

    optional arguments:
      -h, --help            show this help message and exit
      --debug               use debug mode
      --d [D], --dicom [D]  path to directory storing dicom files
      --l [L], --load [L]   path to data directory to load from
      --s [S], --save [S]   path to data directory to save to. Default: L/D
      --ignore_ext          ignore .dcm extension when loading dicoms. Default: False
      --split_by [G]        override dicom tag to split volumes by (eg. `EchoNumbers`)
      --gpu [G]             gpu id. Default: None
      --df [{dicom,nifti}], --data_format [{dicom,nifti}]
                            format to save medical data
      --r2 [T], --r2_threshold [T]
                            r^2 threshold for goodness of fit. Range [0-1).
      --dpi [DPI]           figure resolution in dots per inch (dpi)
      --vf [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}], --visualization_format [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}]
                            format to save figures

    Either `--d` or `---l` must be specified. If both are given, `--d` will be used.


qDESS
--------------------------------------------------------------------------------
Help menu for qDESS.

.. code-block:: bash

    In:
    ---------
    dosma qdess --help

    Out:
    ---------
    usage: DOSMA qdess [-h] [--fc] [--men] [--tc] [--pc]
                       {segment,generate_t2_map,t2} ...

    optional arguments:
      -h, --help            show this help message and exit
      --fc                  analyze femoral cartilage
      --men                 analyze meniscus
      --tc                  analyze tibial cartilage
      --pc                  analyze patellar cartilage

    subcommands:
      qdess subcommands

      {segment,generate_t2_map,t2}
        segment              generate automatic segmentation
        generate_t2_map (t2) generate T2 map


Segmentation
^^^^^^^^^^^^
Automatically segment tissues.

.. code-block:: bash

    In:
    ---------
    dosma qdess segment --help

    Out:
    ---------
    usage: DOSMA qdess segment [-h] --weights_dir WEIGHTS_DIR
                               [--model [{oai-unet2d}]] [--batch_size [B]] [--rms]

    optional arguments:
      -h, --help            show this help message and exit
      --weights_dir WEIGHTS_DIR
                            path to directory with weights
      --model [{oai-unet2d}]
                            Model to use for segmentation. Choices: ['oai-unet2d']
      --batch_size [B]      batch size for inference. Default: 16
      --rms, --use_rms      use root mean square (rms) of two echos for
                            segmentation. Default: False

|T2| Estimation
^^^^^^^^^^^^^^^
Generate |T2| maps using two echos.

.. code-block:: bash

    In:
    ---------
    dosma qdess generate_t2_map --help

    Out:
    ---------
    usage: DOSMA qdess generate_t2_map [-h] [--suppress_fat] [--suppress_fluid]
                                   [--beta [BETA]] [--gl_area [GL_AREA]]
                                   [--tg [TG]]

    optional arguments:
      -h, --help           show this help message and exit
      --suppress_fat       suppress computation on low SNR fat regions. Default:
                           False
      --suppress_fluid     suppress computation on fluid regions. Default: False
      --beta [BETA]        constant for calculating fluid-nulled image
                           (S1-beta*S2). Default: 1.2
      --gl_area [GL_AREA]  GL Area. Defaults to value in dicom tag '0x001910b6'.
      --tg [TG]            Gradient time (in microseconds). Defaults to value in
                           dicom tag '0x001910b7'


CubeQuant
--------------------------------------------------------------------------------
Help menu for CubeQuant.

.. code-block:: bash

    In:
    ---------
    dosma cubequant --help

    Out:
    ---------
    usage: DOSMA cubequant [-h] [--fc] [--men] [--tc] [--pc]
                           {interregister,generate_t1_rho_map,t1_rho} ...

    optional arguments:
      -h, --help            show this help message and exit
      --fc                  analyze femoral cartilage
      --men                 analyze meniscus
      --tc                  analyze tibial cartilage
      --pc                  analyze patellar cartilage

    subcommands:
      cubequant subcommands

      {interregister,generate_t1_rho_map,t1_rho}
        interregister       register to another scan
        generate_t1_rho_map (t1_rho)
                            generate T1-rho map

Interregister
^^^^^^^^^^^^^
Register CubeQuant scan to a target scan. Currently optimized for registering to qDESS target.

.. code-block:: bash

    In:
    ---------
    dosma cubequant interregister --help

    Out:
    ---------
    usage: DOSMA cubequant interregister [-h] --tp TARGET_PATH
                                         [--tm [TARGET_MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --tp TARGET_PATH, --target TARGET_PATH, --target_path TARGET_PATH
                            path to target image in nifti format (.nii.gz)
      --tm [TARGET_MASK_PATH], --target_mask [TARGET_MASK_PATH], --target_mask_path [TARGET_MASK_PATH]
                            path to target mask in nifti format (.nii.gz).
                            Default: None

|T1rho| Estimation
^^^^^^^^^^^^^^^^^^
Compute |T1rho| map using mono-exponential fitting.

.. code-block:: bash

    In:
    ---------
    dosma cubequant generate_t1_rho_map --help

    Out:
    ---------
    usage: DOSMA cubequant generate_t1_rho_map [-h] [--mask_path [MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --mask_path [MASK_PATH]
                            Mask used for fitting select voxels in nifti format
                            (.nii.gz). Default: None

MAPSS
--------------------------------------------------------------------------------
Help menu for MAPSS.

.. code-block:: bash

    In:
    ---------
    dosma mapss --help

    Out:
    ---------
    usage: DOSMA mapss [-h] [--fc] [--men] [--tc] [--pc]
                       {generate_t1_rho_map,t1_rho,generate_t2_map,t2} ...

    optional arguments:
      -h, --help            show this help message and exit
      --fc                  analyze femoral cartilage
      --men                 analyze meniscus
      --tc                  analyze tibial cartilage
      --pc                  analyze patellar cartilage

    subcommands:
      mapss subcommands

      {generate_t1_rho_map,t1_rho,generate_t2_map,t2}
        generate_t1_rho_map (t1_rho)
                            generate T1-rho map using mono-exponential fitting
        generate_t2_map (t2)
                            generate T2 map using mono-exponential fitting

|T1rho| Estimation
^^^^^^^^^^^^^^^^^^^
Compute |T1rho| map using mono-exponential fitting.

.. code-block:: bash

    In:
    ---------
    dosma mapss generate_t1_rho_map --help

    Out:
    ---------
    usage: DOSMA mapss generate_t1_rho_map [-h] [--mask [MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --mask [MASK_PATH], --mp [MASK_PATH], --mask_path [MASK_PATH]
                            mask filepath (.nii.gz) to reduce computational time
                            for fitting. Not required if loading data (ie. `--l`
                            flag) for tissue with mask. Default: None


|T2| Estimation
^^^^^^^^^^^^^^^^^^^
Compute |T2| map using mono-exponential fitting.

.. code-block:: bash

    In:
    ---------
    dosma mapss generate_t2_map --help

    Out:
    ---------
    usage: DOSMA mapss generate_t2_map [-h] [--mask [MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --mask [MASK_PATH], --mp [MASK_PATH], --mask_path [MASK_PATH]
                            mask filepath (.nii.gz) to reduce computational time
                            for fitting. Not required if loading data (ie. `--l`
                            flag) for tissue with mask.. Default: None


UTE Cones
--------------------------------------------------------------------------------
Help menu for UTE Cones.

.. code-block:: bash

    In:
    ---------
    dosma cubequant --help

    Out:
    ---------
    usage: DOSMA cones [-h] [--fc] [--men] [--tc] [--pc]
                       {interregister,generate_t2_star_map,t2_star} ...

    optional arguments:
      -h, --help            show this help message and exit
      --fc                  analyze femoral cartilage
      --men                 analyze meniscus
      --tc                  analyze tibial cartilage
      --pc                  analyze patellar cartilage

    subcommands:
      cones subcommands

      {interregister,generate_t2_star_map,t2_star}
        interregister       register to another scan
        generate_t2_star_map (t2_star)
                            generate T2-star map

Interregister
^^^^^^^^^^^^^
Register Cones scan to a target scan. Currently optimized for registering to qDESS target.

.. code-block:: bash

    In:
    ---------
    dosma cones interregister --help

    Out:
    ---------
    usage: DOSMA cones interregister [-h] --tp TARGET_PATH
                                         [--tm [TARGET_MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --tp TARGET_PATH, --target TARGET_PATH, --target_path TARGET_PATH
                            path to target image in nifti format (.nii.gz)
      --tm [TARGET_MASK_PATH], --target_mask [TARGET_MASK_PATH], --target_mask_path [TARGET_MASK_PATH]
                            path to target mask in nifti format (.nii.gz).
                            Default: None

|T2star| Estimation
^^^^^^^^^^^^^^^^^^^
Compute |T2star| map using mono-exponential fitting.

.. code-block:: bash

    In:
    ---------
    dosma cones generate_t2_star_map --help

    Out:
    ---------
    usage: DOSMA cones generate_t2_star_map [-h] [--mask_path [MASK_PATH]]

    optional arguments:
      -h, --help            show this help message and exit
      --mask_path [MASK_PATH]
                            Mask used for fitting select voxels - in nifti format
                            (.nii.gz). Default: None


### MSK Knee
```
usage: DOSMA knee [-h] [--ml] [--pid [PID]] [--fc] [--men] [--tc] [--t2]
                  [--t1_rho] [--t2_star]

optional arguments:
  -h, --help   show this help message and exit
  --ml         defines slices in sagittal direction going from medial ->
               lateral
  --pid [PID]  specify pid
  --fc         analyze femoral cartilage
  --men        analyze meniscus
  --tc         analyze tibial cartilage
  --t2         quantify t2
  --t1_rho     quantify t1_rho
  --t2_star    quantify t2_star
```

If no quantitative value flag (e.g. `--t2`, `--t1_rho`, `--t2_star`) is specified, all quantitative values will be calculated by default.

If no tissue flag (e.g. `--fc`) is specified, all tissues will be calculated by default.


.. Substitutions
.. |T2| replace:: T\ :sub:`2`
.. |T1| replace:: T\ :sub:`1`
.. |T1rho| replace:: T\ :sub:`1`:math:`{\rho}`
.. |T2star| replace:: T\ :sub:`2`:sup:`*`