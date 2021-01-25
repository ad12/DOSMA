.. _seg_models:

Segmentation Models (dosma.models)
================================================================================
DOSMA currently supports pre-trained models for segmenting, each described in detail below.
Model aliases are string fields used to distinguish/specify particular models in DOSMA (command-line
argument :code:`--model`).

All models are open-sourced under the GNU General Public License v3.0 license.
If you use these models, please reference both DOSMA and the original work.

.. automodule::
   dosma.models

.. autosummary::
   :nosignatures:

   dosma.models.OAIUnet2D
   dosma.models.IWOAIOAIUnet2D
   dosma.models.IWOAIOAIUnet2DNormalized


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
