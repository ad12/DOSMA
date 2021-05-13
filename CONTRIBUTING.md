# Contributing to DOSMA

Thank you for considering to contribute to `DOSMA` and encouraging open-source tools for medical image analysis.

If you use DOSMA in your work (research, company, etc.) and find it useful, spread the word!

This guide is inspired by [Huggingface transformers](https://github.com/huggingface/transformers).

## How to contribute
There are many ways to contribute to DOSMA:

* Issues: Submitting bugs or suggesting new features
* Documentation: Adding to the documentation or to the examples 
* Features: Implementing new features or bug fixes
* Community: Answering questions and helping others get started 

## Submitting a new issue or feature request
Please do your best to follow these guidelines when opening an issue. It will make it signficantly easier to give useful feedback and resolve the issue faster.

### Found a bug?
We would very much appreciate if you could **make sure the bug was not already reported** (use the search bar on Github under Issues). If you cannot find you bug, follow the instructions in the [Bug Report](https://github.com/ad12/DOSMA/issues/new/choose) template.

### Have a new algorithm/model?
Great! Please open an issue and provide the following information:

* Short description of the algorithm and a link to the corresponding publication
* Link to implementation if it is open-source
* Link to model weights if they are available (AI models only)

If you are willing to contribute the algorithm yourself, let us know so we can best guide you.

### Want a new feature (that is not an algorithm)?

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear about it!
  * Is it something you worked on and think could benefit the community? Awesome! Tell us what problem it solved for you.
2. Write a full paragraph describing the feature;
3. Provide a code snippet that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you post it. Follow the instructions in the [Feature Request](https://github.com/ad12/DOSMA/issues/new/choose)

## Contributing
Before writing code, we strongly advise you to search through the existing PRs or issues to make sure that nobody is already working on the same thing. If you are unsure, it is always a good idea to open an issue to get some feedback.

You will need basic git proficiency to be able to contribute to dosma. git is not the easiest tool to use but it has the greatest manual. Type git --help in a shell and enjoy. If you prefer books, [Pro Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [`repository`](https://github.com/ad12/DOSMA) by clicking on the 'Fork' button the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/DOSMA.git
   $ cd DOSMA
   $ git remote add upstream https://github.com/ad12/DOSMA.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

  ```bash
  pip install -e ".[dev]"
  ```

5. Develop features on your branch.

    As you work on the features, you should make sure that the test suite passes:

    ```bash
    $ make test
    ```

    After you make changes, autoformat them with:

    ```bash
    $ make autoformat
    ```


    If you modify documentation (`docs/source`), verify the documents build:

    ```bash
    $ make build-docs
    ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

HINT: Run all major formatting and checks using the following:

```bash
make autoformat test build-docs
```

### Checklist

1. Make the title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, mention the issue number in
  the pull request description
3. If your PR is a work in progress, start the title with `[WIP]`
4. Make sure existing tests pass;
5. Add high-coverage tests. Additions without tests will not be merged
6. All public methods must have informative docstrings in the google style.

### Tests

Library tests can be found in the 
[tests folder](https://github.com/ad12/DOSMA/tree/master/tests).

From the root of the repository, here's how to run tests with `pytest` for the library:

```bash
$ make test
```

### Style guide
`dosma` follows the [google style](https://google.github.io/styleguide/pyguide.html) for documentation.


## Popular Interfaces
DOSMA offers a range of interfaces to help developers get started with open-sourcing their algorithms.
Two interfaces that have been increasingly popular are

1. `ScanSequences`: An interface for implementing scan-specific algorithms (e.g. quantitative MRI)
2. `AI toolbox`: An interface for distributing and using pretrained models.

**NOTE**: The guides below are slightly outdated. Please reach out to us for up-to-date instructions on how to proceed.

### Scan Sequences
Thank you for adding analysis support for new scan sequences! Please follow the instructions below to streamline adding and using these sequences.

1. Use/create a unique, but relevant, name for your sequence.
    - `qDESS`: Quantitative Double Echo in Steady-State
    - `MAPSS`: Magnetization-Prepared Angle-Modulated Partitioned k-Space Spoiled Gradient Echo Snapshots
2. Create a new file in the `scan_sequences` folder with your sequence name (use [snake_casing](https://en.wikipedia.org/wiki/Snake_case))
3. Create a class in the file that inherits from `dosma.scan_sequences.ScanSequence`.

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

### Automatic Segmentation
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
