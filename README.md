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

### Anatomy
Analysis for the following anatomical regions are supported

#### MSK - knee

**Tissues**:
- Femoral Cartilage

## Installation
Download this repo to your disk.

For pretrained weights for MSK knee segmentation, request access to this [Google Drive](https://drive.google.com/drive/u/0/folders/1VtVzOAS6VbFzpEi9Fivy6BgcMubfFlL-). Note that these weights are optimized to run on single-echo RMS DESS sequence as used in the [OA initiative](https://oai.epi-ucsf.org/datarelease/).

Save these weights in an accessible location. **Do not rename these files**.

## Shell interface help
To run the program from a shell, run `python -m opt/path/pipeline` with the flags detailed below. `opt/path` is the path to the file `python`

### Base information

```
usage: pipeline [-h] [-d [D]] [-l [L]] [-s [S]] [-e [E]] [--gpu [G]]
            {dess,cubequant,cq,knee} ...

Pipeline for segmenting MRI knee volumes

positional arguments:
  {dess,cubequant,cq,knee}
                        sub-command help
    dess                analyze DESS sequence
    cubequant (cq)      analyze cubequant sequence
    knee                analyze tissues

optional arguments:
  -h, --help            show this help message and exit
  -d [D], --dicom [D]   path to directory storing dicom files
  -l [L], --load [L]    path to data directory to load from
  -s [S], --save [S]    path to directory to save mask. Default: D/L
  -e [E], --ext [E]     extension of dicom files. Default 'dcm'
  --gpu [G]             gpu id
```

### DESS
```
usage: pipeline dess [-h] [-rms] [-t2] {segment} ...

positional arguments:
  {segment}   sub-command help

optional arguments:
  -h, --help  show this help message and exit
  -rms        use rms for segmentation
  -t2         compute T2 map
```

#### Segmentation
```
usage: pipeline dess segment [-h] [--model [{unet2d}]] [--weights_dir WEIGHTS_DIR]
                         [--batch_size [B]] [-fc]

optional arguments:
  -h, --help            show this help message and exit
  --model [{unet2d}]
  --weights_dir WEIGHTS_DIR
                        path to directory with weights
  --batch_size [B]      batch size for inference. Default: 32
  -fc                   handle femoral cartilage
```

### MSK Knee
```
usage: pipeline knee [-h] [--orientation [{RIGHT,LEFT}]] [-fc] [-t2 T2]
                     [-t1_rho T1_RHO] [-t2_star T2_STAR]

optional arguments:
  -h, --help            show this help message and exit
  --orientation [{RIGHT,LEFT}]
                        knee orientation (left or right)
  -fc                   analyze femoral cartilage
  -t2 T2                quantify t2
  -t1_rho T1_RHO        quantify t1_rho
  -t2_star T2_STAR      quantify t2_star
```

## Use cases

We detail use cases that could be useful for analyzing data. We assume that all scans are stored per patient, meaning that the folder structure looks like below:

```
research_data
    | patient01
        | dess
            | I001.dcm
            | I002.dcm
            | I003.dcm
            ....
        | cubequant
        | cones
        | <OTHER SCAN SEQUENCE DATA>
    | patient02
    | patient03
    | unet_weights
    ...
```

All use cases assume that the [current working directory](https://www.computerhope.com/jargon/c/currentd.htm) is this repo. If the working directory is different, make sure to specify the path to ```pipeline.py``` when running the script. For example, ```python -m ~/MyRepo/pipeline.py``` if the repo is located in the user directory.

### DESS
#### Case 1
*Analyze patient01 T<sub>2</sub> using DESS data*

1. Calculate 3D t2 map
```
python -m pipeline -d research_data/patient01/dess -s research_data/patient01/data dess -t2
```
2. Segment femoral cartilage using RMS of two echo dess echos
```
python -m pipeline -d research_data/patient01/dess -s research_data/patient01/data dess -rms segment --weights_dir unet_weights
```

3. Calculate T_<sub>2</sub> time for femoral cartilage
```
python -m pipeline -l research_data/patient01/data -s research_data/patient01/data knee -fc -t2
```
