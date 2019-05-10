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
