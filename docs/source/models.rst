.. _seg_models:

Models (dosma.models)
================================================================================
DOSMA currently supports pre-trained deep learning models for segmenting, each described in detail below.
Model aliases are string fields used to distinguish/specify particular models in DOSMA (command-line
argument :code:`--model`).

All models are open-sourced under the GNU General Public License v3.0 license.
If you use these models, please reference both DOSMA and the original work.

.. automodule::
   dosma.models

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.models.OAIUnet2D
   dosma.models.IWOAIOAIUnet2D
   dosma.models.IWOAIOAIUnet2DNormalized
   dosma.models.StanfordQDessUNet2D


OAI 2D U-Net
--------------------------------------------------------------------------------
A 2D U-Net trained on a downsampled rendition of the OAI iMorphics DESS dataset :cite:`chaudhari2018open`.
Inputs are zero-mean, unit standard deviation normalized before segmentation.

Aliases: :code:`oai-unet2d`, :code:`oai_unet2d`


IWOAI Segmentation Challenge - Team 6 2D U-Net
--------------------------------------------------------------------------------
This model was submitted by Team 6 to the 2019 International Workshop on Osteoarthritis Segmentation
:cite:`desai2020international`.
It consists of a 2D U-Net trained on the standardized OAI training dataset.

Note, inputs are not normalized before segmentation and therefore may be difficult to generalize to
DESS scans with different parameters than the OAI.

Aliases: :code:`iwoai-2019-t6`


IWOAI Segmentation Challenge - Team 6 2D U-Net (Normalized)
--------------------------------------------------------------------------------
This model is a duplicate of the `iwoai-2019-t6` network (above), but differs in that it uses
zero-mean, unit standard deviation normalized inputs. This may make the network more robust to
different DESS scan parameters and/or scanner vendors.

While this model was not submitted to the IWOAI challenge, the architecture, training parameters, and dataset are
identical to the Team 6 submission. Performance on the standardized OAI test set was similar to the original network
submitted by Team 6 (see table below).

Aliases: :code:`iwoai-2019-t6-normalized`

.. table:: Average (standard deviation) performance summary on OAI test set.
           Coefficient of variation is calculated as root-mean-square value.

    =========  ===================  ==================  ====================  ===============
    ..         Femoral Cartilage    Tibial Cartilage    Patellar Cartilage    Meniscus
    =========  ===================  ==================  ====================  ===============
    Dice       0.906 +/- 0.014      0.881 +/- 0.033     0.857 +/- 0.080       0.870 +/- 0.032
    VOE        0.171 +/- 0.023      0.211 +/- 0.052     0.242 +/- 0.108       0.229 +/- 0.049
    RMS-CV     0.019 +/- 0.011      0.048 +/- 0.029     0.076 +/- 0.061       0.045 +/- 0.025
    ASSD (mm)  0.174 +/- 0.020      0.270 +/- 0.166     0.243 +/- 0.106       0.344 +/- 0.111
    =========  ===================  ==================  ====================  ===============


SKM-TEA qDESS Knee Segmentation - 2D U-net
--------------------------------------------------------------------------------
This collection of models are trained on the `SKM-TEA dataset <https://github.com/StanfordMIMI/skm-tea>`_
(previously known as the *2021 Stanford qDESS Knee Dataset*).
Details of the different models that are trained are shown in the training configurations
distributed with the weights.


   *  ``qDESS_2021_v1-rms-unet2d-pc_fc_tc_men_weights.h5``: This is the baseline
      RSS model trained on the SKM-TEA v1 dataset.
      Though the same hyperparameters were used, this model (trained with Tensorflow/Keras)
      performs better than the PyTorch implementation specified in the main paper.
      Results are shown in the table below.
   *  ``qDESS_2021_v0_0_1-rms-pc_fc_tc_men_weights.h5``: This model is trained on the
      2021 Stanford qDESS knee dataset (v0.0.1).
   *  ``qDESS_2021_v0_0_1-traintest-rms-pc_fc_tc_men_weights.h5``: This model
      is trained on both the train and test set of the 2021 Stanford qDESS knee
      dataset (v0.0.1).

Aliases: :code:`stanford-qdess-2021-unet2d`, :code:`skm-tea-unet2d`


.. table:: Mean +/- standard deviation performance summary on SKM-TEA v1 dataset.

   =========  ===================  ==================  ====================  ===============
   ..         Femoral Cartilage    Tibial Cartilage    Patellar Cartilage    Meniscus
   =========  ===================  ==================  ====================  ===============
   Dice       0.882 +/- 0.033      0.865 +/- 0.035     0.879 +/- 0.103       0.847 +/- 0.068
   VOE        0.210 +/- 0.052      0.237 +/- 0.053     0.205 +/- 0.121       0.261 +/- 0.092
   CV         0.051 +/- 0.033      0.053 +/- 0.037     0.049 +/- 0.077       0.052 +/- 0.052
   ASSD (mm)  0.265 +/- 0.114      0.354 +/- 0.250     0.477 +/- 0.720       0.485 +/- 0.307
   =========  ===================  ==================  ====================  ===============