# DOSMA: Deep Open-Source Medical Image Analysis
[![Documentation Status](https://readthedocs.org/projects/dosma/badge/?version=stable)](https://dosma.readthedocs.io/en/stable/?badge=stable)

DOSMA is an AI-powered Python library for medical image analysis. This includes, but is not limited to:
- image processing (denoising, super-resolution, registration, segmentation, etc.)
- quantitative fitting and image analysis
- anatomical visualization and analysis (patellar tilt, femoral cartilage thickness, etc.)

We hope that this open-source pipeline will be useful for quick anatomy/pathology analysis from MRI and will serve as a hub for adding support for analyzing different anatomies and scan sequences.

[Documentation](http://dosma.readthedocs.io/) | [Questionnaire](https://forms.gle/sprthTC2swyt8dDb6)

## Installation
DOSMA requires Python 3.6+. The core module depends on numpy, nibabel, nipype,
pandas, pydicom, scikit-image, scipy, PyYAML, and tqdm.

Additional AI features can be unlocked by installing tensorflow and keras. To
enable built-in registration functionality, download [elastix](https://elastix.lumc.nl/download.php).
Details can be found in the [setup documentation](https://ad12.github.io/DOSMA/build/html/general/installation.html#setup).

To install DOSMA, run:

```bash
pip install dosma
```

If you would like to contribute to DOSMA, we recommend you clone the repository and
install DOSMA with `pip` in editable mode.

```bash
git clone git@github.com:ad12/DOSMA.git
cd DOSMA
pip install -e '.[dev]'
make dev
```

To run tests, build documentation and contribute, run
```bash
make autoformat test build-docs
```

## How to Cite
```
@inproceedings{desai2019dosma,
   Title={DOSMA: A deep-learning, open-source framework for musculoskeletal MRI analysis.},
   Author =  {Desai, Arjun D and Barbieri, Marco and Mazzoli, Valentina and Rubin, Elka and Black, Marianne S and Watkins, Lauren E and Gold, Garry E and Hargreaves, Brian A and Chaudhari, Akshay S},
   Booktitle={Proc. Intl. Soc. Mag. Reson. Med},
   Volume={27},
   Number={1106},
   Year={2019}
}
```

In addition to DOSMA, please also consider citing the work that introduced the method used for analysis.
