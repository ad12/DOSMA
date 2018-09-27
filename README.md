# MRI Knee Tissue Segmentation

This pipeline is an open-source pipeline for MRI image segmentation, registration, and quantitative analysis.

## Installation
Download this repo to a location on your disk. 
Request for pretrained weights from this [Google Drive](https://drive.google.com/drive/u/0/folders/1VtVzOAS6VbFzpEi9Fivy6BgcMubfFlL-). 
Save these weights in an accessible location. **Do not rename these files**.

## Shell interface
To run the program from a shell, run `python -m opt/path/pipeline` with the flags detailed below. `opt/path` is the path to the file `python`

Note this program is meant to be run from the command line. As a results, all import statements are local imports.


#### Flags
Required:
- ```-d D, --dicom D```: directory storing dicom files (referred to as '_dicom directory_')

Optional:
- ``-s [S], --save [S]``: directory to save mask. Default is dicom directory
- `-e [E], --ext [E]`: dicom file extension. Default is no extension. This means that if this flag is not specified, all files in the dicom directory will be read as dicom files. In this case, please remove any non dicom files from that directory to avoid errors
- `-f`: segment femoral cartilage
- `-t`: segment tibial cartilage
- `-m`: segment meniscus
- `-p`: segment patellar cartilage

If none of the segmentation flags (`-f`, `-t`, `-m`. or `-p`) are specified, all tissue will be segmented.

#### Help
```
usage: generate_mask.py [-h] [-d D] [-s [S]] [-e [E]] [-f] [-t] [-m] [-p]

Segment MRI knee volumes using ARCHITECTURE (ADD SOURCE)

optional arguments:
  -h, --help          show this help message and exit
  -d D, --dicom D     path to directory storing dicom files
  -s [S], --save [S]  path to directory to save mask. Default: D
  -e [E], --ext [E]   extension of dicom files
  -f                  segment femoral cartilage
  -t                  segment tibial cartilage
  -m                  segment meniscus
  -p                  segment patellar cartilage

NOTE: by default all tissues will be segmented unless specific flags are
provided
```

### Output
The outputs are binarized (0/1) uint8 3D tiff images. The names of the files will correspond to the tissue as follows:

- Femoral cartilage: `fc.tiff`
- Tibial cartilage: `tc.tiff`
- Meniscus: `men.tiff`
- Patellar cartilage: `pc.tiff`


### Usage

All use cases assume that the [current working directory](https://www.computerhope.com/jargon/c/currentd.htm) is this repo. If the working directory is different, make sure to specify the path to ```generate_mask.py``` when running the script. For example, ```python -m ~/MyRepo/generate_mask.py``` if the repo is located in the user directory.

#### Case 1
###### Read dicoms from ```~/path_to_dicom_directory``` and store masks in same directory:

`python -m generate_mask.py -d ~/path_to_dicom_directory`

#### Case 2
###### Read dicoms and store mask in different directory:

`python -m generate_mask.py -d ~/path_to_dicom_directory -s ~/path_to_save_directory`

#### Case 3
###### Read dicoms with extension `.dcm`:

`python -m generate_mask.py -d ~/path_to_dicom_directory -e .dcm`

#### Case 4
###### Only segment femoral and patellar cartilage:

`python -m generate_mask.py -d ~/path_to_dicom_directory -fp`

###### Only segment tibial cartilage:

`python -m generate_mask.py -d ~/path_to_dicom_directory -t`

###### Segment all tissues:

`python -m generate_mask.py -d ~/path_to_dicom_directory`

OR

`python -m generate_mask.py -d ~/path_to_dicom_directory -ftpm`

## Dependencies
- python >= 3.5.5
- tensorflow >= 1.6.0
- keras >= 2.1.6
- natsort >= 5.3.3
- numpy >= 1.14.3
- pydicom >= 1.1.0
- SimpleITK >= 1.1.0
