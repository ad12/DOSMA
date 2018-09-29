# Open-Source MRI Pipeline

This pipeline is an open-source pipeline for MRI image segmentation, registration, and quantitative analysis.

The current code uses the [command line interface](https://www.computerhope.com/jargon/c/commandi.htm) for use. Pull requests for a GUI to command-line translation are welcome.

## Supported Features

### Scans
The following scan sequences are supported. All sequences with multiple echos, spin_lock_times, etc. should have metadata in the dicom header specifying this information.

#### Double echo DESS

##### Data format
All data should be provided in the dicom format. Currently only sagittal orientation dicoms are supported.

Dicom files should be named in the format *001.dcm: echo1*, *002.dcm: echo2*, *003.dcm: echo1*, etc.

##### Quantitative Values
*T<sub>2</sub>*: Calculate T<sub>2</sub> map using dual echos

##### Actions
*Segmentation*

##### Command-line help

### Anatomy
Analysis for the following anatomical regions are supported

#### MSK - knee

**Tissues**:
- Femoral Cartilage

## Installation
Download this repo to your disk.

For pretrained weights for MSK - knee segmentation, request access to this [Google Drive](https://drive.google.com/drive/u/0/folders/1VtVzOAS6VbFzpEi9Fivy6BgcMubfFlL-).
Save these weights in an accessible location. **Do not rename these files**.

## Shell interface
To run the program from a shell, run `python -m opt/path/pipeline` with the flags detailed below. `opt/path` is the path to the file `python`

### Usage

All use cases assume that the [current working directory](https://www.computerhope.com/jargon/c/currentd.htm) is this repo. If the working directory is different, make sure to specify the path to ```pipeline.py``` when running the script. For example, ```python -m ~/MyRepo/pipeline.py``` if the repo is located in the user directory.

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
