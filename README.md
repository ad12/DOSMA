# DOSMA: Deep Open-Source Medical Image Analysis
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ad12/DOSMA/CI)
[![codecov](https://codecov.io/gh/ad12/DOSMA/branch/master/graph/badge.svg?token=X2FRQJHV2M)](https://codecov.io/gh/ad12/DOSMA)
[![Documentation Status](https://readthedocs.org/projects/dosma/badge/?version=latest)](https://dosma.readthedocs.io/en/latest/?badge=latest)

[Documentation](http://dosma.readthedocs.io/) | [Questionnaire](https://forms.gle/sprthTC2swyt8dDb6) | [DOSMA Basics Tutorial](https://colab.research.google.com/drive/1zY5-3ZyTBrn7hoGE5lH0IoQqBzumzP1i?usp=sharing)

DOSMA is an AI-powered Python library for medical image analysis. This includes, but is not limited to:
- image processing (denoising, super-resolution, registration, segmentation, etc.)
- quantitative fitting and image analysis
- anatomical visualization and analysis (patellar tilt, femoral cartilage thickness, etc.)

We hope that this open-source pipeline will be useful for quick anatomy/pathology analysis and will serve as a hub for adding support for analyzing different anatomies and scan sequences.

## Installation
DOSMA requires Python 3.7+. The core module depends on numpy, nibabel, nipype,
pandas, pydicom, scikit-image, scipy, PyYAML, and tqdm.

Additional AI features can be unlocked by installing tensorflow and keras. To
enable built-in registration functionality, download [elastix](https://elastix.lumc.nl/download.php).
Details can be found in the [setup documentation](https://dosma.readthedocs.io/en/latest/general/installation.html#setup).

To install DOSMA, run:

```bash
pip install dosma

# To install with AI support
pip install dosma[ai]
```

If you would like to install from source - to get the latest versions, or to build for your specific system: 

If you would like to install from source then use: 

```bash
git clone git@github.com:ad12/DOSMA.git
cd DOSMA
pip install .
```

To get AI support: 

```bash
git clone git@github.com:ad12/DOSMA.git
cd DOSMA
pip install '.[ai]'
```

If you would like to contribute to DOSMA, we recommend you clone the repository and
install DOSMA with `pip` in editable mode.

```bash
git clone git@github.com:ad12/DOSMA.git
cd DOSMA
pip install -e '.[dev,docs]'
make dev
```

To run tests, build documentation and contribute, run
- If Elastix not installed, must indicate this to system by running the following on the commandline: 
   `export "DOSMA_UNITTEST_DISABLE_ELASTIX=True"`
- If test data not available add:
   `export ""DOSMA_UNITTEST_DISABLE_DATA"=True"`
```bash
make autoformat test build-docs
```

## Features
### Simplified, Efficient I/O
DOSMA provides efficient readers for DICOM and NIfTI formats built on nibabel and pydicom. Multi-slice DICOM data can be loaded in
parallel with multiple workers and structured into the appropriate 3D volume(s). For example, multi-echo and dynamic contrast-enhanced (DCE) MRI scans have multiple volumes acquired at different echo times and trigger times, respectively. These can be loaded into multiple volumes with ease:

```python
import dosma as dm

multi_echo_scan = dm.load("/path/to/multi-echo/scan", group_by="EchoNumbers", num_workers=8, verbose=True)
dce_scan = dm.load("/path/to/dce/scan", group_by="TriggerTime")
```

### Data-Embedded Medical Images
DOSMA's [MedicalVolume](https://dosma.readthedocs.io/en/latest/generated/dosma.MedicalVolume.html#dosma.MedicalVolume) data structure supports array-like operations (arithmetic, slicing, etc.) on medical images while preserving spatial attributes and accompanying metadata. This structure supports NumPy interoperability, intelligent reformatting, fast low-level computations, and native GPU support. For example, given MedicalVolumes `mvA` and `mvB` we can do the following:

```python
# Reformat image into Superior->Inferior, Anterior->Posterior, Left->Right directions.
mvA = mvA.reformat(("SI", "AP", "LR"))

# Get and set metadata
study_description = mvA.get_metadata("StudyDescription")
mvA.set_metadata("StudyDescription", "A sample study")

# Perform NumPy operations like you would on image data.
rss = np.sqrt(mvA**2 + mvB**2)

# Move to GPU 0 for CuPy operations
mv_gpu = mvA.to(dosma.Device(0))

# Take slices. Metadata will be sliced appropriately.
mv_subvolume = mvA[10:20, 10:20, 4:6]
```

### Built-in AI Models
DOSMA is built to be a hub for machine/deep learning models. A complete list of models and corresponding publications can be found [here](https://dosma.readthedocs.io/en/latest/models.html).
We can use one of the knee segmentation models to segment a MedicalVolume `mv` and model
`weights` [downloaded locally](https://dosma.readthedocs.io/en/latest/installation.html#segmentation):

```python
from dosma.models import IWOAIOAIUnet2DNormalized

# Reformat such that sagittal plane is last dimension.
mv = mv.reformat(("SI", "AP", "LR"))

# Do segmentation
model = IWOAIOAIUnet2DNormalized(input_shape=mv.shape[:2] + (1,), weights_path=weights)
masks = model.generate_mask(mv)
```

### Parallelizable Operations
DOSMA supports parallelization for compute-heavy operations, like curve fitting and image registration.
Image registration is supported thru the [elastix/transformix](https://elastix.lumc.nl/download.php) libraries. For example we can use multiple workers to register volumes to a target, and use the registered outputs for per-voxel monoexponential fitting:

```python
# Register images mvA, mvB, mvC to target image mv_tgt in parallel
_, (mvA_reg, mvB_reg, mvC_reg) = dosma.register(
   mv_tgt,
   moving=[mvA, mvB, mvC],
   parameters="/path/to/elastix/registration/file",
   num_workers=3,
   return_volumes=True,
   show_pbar=True,
)

# Perform monoexponential fitting.
def monoexponential(x, a, b):
   return a * np.exp(b*x)

fitter = dosma.CurveFitter(
   monoexponential,
   num_workers=4,
   p0={"a": 1.0, "b": -1/30},
)
popt, r2 = fitter.fit(x=[1, 2, 3, 4], [mv_tgt, mvA_reg, mvB_reg, mvC_reg])
a_fit, b_fit = popt[..., 0], popt[..., 1]
```

## Citation
```
@inproceedings{desai2019dosma,
  title={DOSMA: A deep-learning, open-source framework for musculoskeletal MRI analysis},
  author={Desai, Arjun D and Barbieri, Marco and Mazzoli, Valentina and Rubin, Elka and Black, Marianne S and Watkins, Lauren E and Gold, Garry E and Hargreaves, Brian A and Chaudhari, Akshay S},
  booktitle={Proc 27th Annual Meeting ISMRM, Montreal},
  pages={1135},
  year={2019}
}
```

In addition to DOSMA, please also consider citing the work that introduced the method used for analysis.
