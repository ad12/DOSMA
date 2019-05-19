# Contributing to DOSMA

Thank you for considering to contribute to `DOSMA` and encouraging open-source tools for MRI analysis.

Please read this guide carefully to avoid unnecessary coding overhead and to deploy your additions quickly.

## Table of Contents
<!--ts-->
   * [Getting Started](#getting-started)
   * [Scan Sequences](#scan-sequences)
      * [Separating Sequences Into Volumes](#separating-sequences-into-volumes)
      * [Fitting Quantitative Values](#fitting-quantitative-values)
      * [Scan Template](#scan-template)
   * [Automatic Segmentation](#automatic-segmentation)
      * [Deep-Learning Segmentation](#deep-learning-segmentation)
      * [Naming Weight Files](#naming-weight-files)
      * [Sharing Files](#sharing-files)
      * [Segmentation Model Template](#segmentation-model-template)
        * [Keras (Tensorflow)](#Keras (Tensorflow)
<!--te-->

## Getting Started
Basic Workflow:
1. Fork the repository
2. `git clone` the forked repository to your machine
3. `cd` to where you have cloned the repository
4. Use `conda` to install dependencies from the `envs/environment_<OS>.yml` file
5. Make changes and corresponding unit-tests
6. Submit pull request (PR)

**All PRs made without sufficient unit-tests will be rejected.**

## Scan Sequences
Thank you for adding analysis support for new scan sequences! Please follow the instructions below to streamline adding and using these sequences.

1. Use/create a unique, but relevant, name for your sequence.
    - `qDESS`: Quantitative Double Echo in Steady-State
    - `MAPSS`: Magnetization-Prepared Angle-Modulated Partitioned k-Space Spoiled Gradient Echo Snapshots
2. Create a new file in the `scan_sequences` folder with your sequence name (use [snake_casing](https://en.wikipedia.org/wiki/Snake_case))
3. Create a class in the file that inherits from `ScanSequence`, `TargetSequence`, or `NonTargetSequence`.

#### Separating Sequences Into Volumes
In many quantitative sequences, multiple echos are acquired for each slice to perform some form of voxel-wise quantitative fitting/extrapolation. We define a **volume** as a 3D matrix with values from a single echo. Therefore, a *qDESS* sequence, which has two echos, has volumes.

Each scan sequence implementation has a instance variable called `volumes`, in which the total pool of DICOM files are intelligently split into their respective volumes. For *qDESS*, the volumes instance variable would be a list with `len(volumes) = 2`. Sequences encoding for one echo will have the `volumes` field be a list with `len(volumes) = 1`.

By default, all scan sequences are split by the `EchoNumbers` DICOM tag, which specifies which echo the current DICOM slice corresponds to. However, depending on the scan sequence, where the volumes may need to be split by a different DICOM tag, override the field `__DEFAULT_SPLIT_BY__` in the scan sequence class.

#### Fitting quantitative values
Any scans that support quantitative parameter fitting should have a method named `generate_<QuantitativeValue>_map` (e.g. `generate_t1_rho_map`).

```python
def generate_<QuantitativeValue>_map(self, tissue: Tissue, ...) --> QuantitativeValue:
  ...
```

#### Scan Template
This template defines the basic implementation for any new scan sequence
```python
class NewScanSequence(ScanSequence/TargetSequence/NonTargetSequence):
  NAME=''  # add name of sequence in snake casing here
  __DEFAULT_SPLIT_BY__ = 'EchoNumbers'  # specify dicom tag to split volume by. default: 'EchoNumbers`

  def __validate_scan__(self) -> bool:
    """Validate this scan (usually done by checking dicom header tags, if available)
      :return a boolean
    """
```

## Automatic Segmentation
Robust automatic segmentation methods are critical to eliminating the bottleneck for morphological and quantitative analysis. The DOSMA framework enables easy integration of automatic segmentation techniques.

#### Deep-Learning Segmentation
Typically, deep learning segmentation algorithms consist of four blocks during inference:
1. **Data preprocessing**: Data is typically preprocessed to fit in the distribution expected by the network
    - e.g: zero-mean & unit-standard deviation whitening, scaling, etc.
2. **Architecture**: Each network can have a unique architecture (U-Net, SegNet, etc.). These architectures can be hard-coded into the file itself, or can be loaded from a `JSON` format (as outputted by Keras)
3. **Weights**: The model parameters, which determine the weights and biases for different layers of the network, can be exported to an `h5` file and loaded in dynamically.
4. **Mask Post-processing**: Some post-processing you wish to complete on the probability/binarized output.
  - e.g: Conditional Random Fields (CRFs), etc.

#### Naming Weight Files
All weight files should contain the aliases of the tissues that they can segment and must end with the extension `.h5`. For example, a weights file saved for femoral cartilage segmentation should have the alias `fc` in its name (eg. `fc.h5`, `oai-unet-fc.h5`, etc.).

#### Sharing Files
Weight files must be shared. Currently, there is no centralized location where these files can be hosted. As a result, please host the data on the cloud (google drive, box, dropbox, etc) and allow public download.

#### Segmentation Model Template
##### Keras (Tensorflow)
This template defines the basic implementation for any new Keras segmentation model.
See `models/oaiunet2d.py` for an example.
```python
class KerasSegModel(SegModel):
    """
    Abstract wrapper for Keras model used for semantic segmentation
    """

    def __load_keras_model__(self, input_shape):
        """Build Keras architecture

        :param input_shape: tuple or list of tuples for initializing input(s) into Keras model

        :return: a Keras model
        """

    def generate_mask(self, volume: MedicalVolume):
        """Segment the MRI volumes

        :param volume: A Medical Volume (height, width, slices)

        :return: A Medical volume or list of Medical Volumes with volume as binarized (0,1) uint8 3D numpy array of shape volumes.shape

        :raise ValueError if volumes is not 3D numpy array
        :raise ValueError if tissue is not a string or not in list permitted tissues

        """
```
