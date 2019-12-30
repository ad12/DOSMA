# DOSMA: Deep Open-Source MRI Analysis

This repository hosts an open-source Python library for MRI processing techniques. This includes, but is not limited to:
- image processing tasks (denoising, super-resolution, segmentation, etc.)
- relaxation parameter analysis (T1, T1-rho, T2, T2*, etc.)
- anatomical features (patellar tilt, femoral cartilage thickness, etc.)

We hope that this open-source pipeline will be useful for quick anatomy/pathology analysis from MRI and will serve as a hub for adding support for analyzing different anatomies and scan sequences.

## Supported Commands
Currently, this pipeline supports analysis of the femoral cartilage in the knee using [qDESS](https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.26577) and cubequant scanning protocols. Basic cones protocol is provided, but still under construction. Details are provided below.

### Scans
The following scan sequences are supported. All sequences with multiple echos, spin_lock_times, etc. should have metadata in the dicom header specifying this information.

#### Quantitative Double echo steady state (qDESS)

##### Data format
All data should be provided in the dicom format.

Dicom files should be named in the format *001.dcm: echo1*, *002.dcm: echo2*, *003.dcm: echo1*, etc.

### Help Menu (General)
```
In:
---------
dosma -h

Out:
---------
usage: DOSMA [-h] [--debug] [--d [D]] [--l [L]] [--s [S]] [--ignore_ext]
             [--split_by [G]] [--gpu [G]] [--df [{dicom,nifti}]] [--r2 [T]]
             [--dpi [DPI]]
             [--vf [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}]]
             {qdess,cubequant,mapss,knee} ...

A deep-learning powered open source MRI analysis pipeline

positional arguments:
  {qdess,cubequant,mapss,knee}
                        sub-command help
    qdess               analyze qdess sequence
    cubequant           analyze cubequant sequence
    mapss               analyze mapss sequence
    knee                calculate/analyze quantitative data for knee

optional arguments:
  -h, --help            show this help message and exit
  --debug               use debug mode
  --d [D], --dicom [D]  path to directory storing dicom files
  --l [L], --load [L]   path to data directory to load from
  --s [S], --save [S]   path to data directory to save to. Default: L/D
  --ignore_ext          ignore .dcm extension when loading dicoms. Default: False
  --split_by [G]        override dicom tag to split volumes by (eg. `EchoNumbers`)
  --gpu [G]             gpu id. Default: None
  --df [{dicom,nifti}], --data_format [{dicom,nifti}]
                        format to save medical data
  --r2 [T], --r2_threshold [T]
                        r^2 threshold for goodness of fit. Range [0-1).
  --dpi [DPI]           figure resolution in dots per inch (dpi)
  --vf [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}], --visualization_format [{png,eps,pdf,jpeg,pgf,ps,raw,rgba,svg,svgz,tiff}]
                        format to save figures

Either `--d` or `---l` must be specified. If both are given, `-d` will be used
```

### qDESS
The qDESS protocol used here is detailed in [this](https://onlinelibrary.wiley.com/doi/pdf/10.1002/jmri.25883) paper referenced below:

*Chaudhari, Akshay S., et al. "Five‐minute knee MRI for simultaneous morphometry and T2 relaxometry of cartilage and meniscus and for semiquantitative radiological assessment using double‐echo in steady‐state at 3T." JMRI 47.5 (2018): 1328-1341.*


```
In:
---------
dosma qdess -h

Out:
---------
usage: DOSMA qdess [-h] [--fc] [--men] [--tc] [--pc]
                   {segment,generate_t2_map,t2} ...

optional arguments:
  -h, --help            show this help message and exit
  --fc                  analyze femoral cartilage
  --men                 analyze meniscus
  --tc                  analyze tibial cartilage
  --pc                  analyze patellar cartilage

subcommands:
  qdess subcommands

  {segment,generate_t2_map,t2}
    segment              generate automatic segmentation
    generate_t2_map (t2) generate T2 map
```

#### Segmentation
```
In:
---------
dosma qdess segment -h

Out:
---------
usage: DOSMA qdess segment [-h] --weights_dir WEIGHTS_DIR
                           [--model [{oai-unet2d}]] [--batch_size [B]] [--rms]

optional arguments:
  -h, --help                show this help message and exit
  --weights_dir WEIGHTS_DIR path to directory with weights
  --model [{oai-unet2d}]    Model to use for segmentation. Choices: ['oai-unet2d']
  --batch_size [B]          batch size for inference. Default: 16
  --rms, --use_rms          use root mean square (rms) of two echos for segmentation. Default: False
```

#### T2 Estimation
```
In:
---------
dosma qdess t2 -h

Out:
---------
usage: DOSMA qdess generate_t2_map [-h] [--suppress_fat] [--gl_area [GL_AREA]]
                                   [--tg [TG]]

optional arguments:
  -h, --help           show this help message and exit
  --suppress_fat       suppress computation on low SNR fat regions. Default: False
  --gl_area [GL_AREA]  gl_area. Default: None
  --tg [TG]            tg. Default: None
```

### Cubequant
The cubequant protocol used here is detailed below:

```
usage: DOSMA cubequant [-h] [--fc] [--men] [--tc]
                       {interregister,generate_t1_rho_map,t1_rho} ...

optional arguments:
  -h, --help            show this help message and exit
  --fc                  analyze femoral cartilage
  --men                 analyze meniscus
  --tc                  analyze tibial cartilage

subcommands:
  cubequant subcommands

  {interregister,generate_t1_rho_map,t1_rho}
    interregister       register to another scan
    generate_t1_rho_map (t1_rho)
                        generate T1-rho map
```

#### Interregister
Register cubequant scan to a target scan

```
usage: DOSMA cubequant interregister [-h] --tp TARGET_PATH
                                     [--tm [TARGET_MASK_PATH]]

optional arguments:
  -h, --help            show this help message and exit
  --tp TARGET_PATH, --target TARGET_PATH, --target_path TARGET_PATH
                        path to target image in nifti format (.nii.gz)
  --tm [TARGET_MASK_PATH], --target_mask [TARGET_MASK_PATH], --target_mask_path [TARGET_MASK_PATH]
                        path to target mask in nifti format (.nii.gz).
                        Default: None
```

#### Interregister
```
usage: pipeline cones interregister [-h] [-ts TS] [-tm [TM]]

optional arguments:
  -h, --help  show this help message and exit
  -ts TS      path to target image. Type: nifti (.nii.gz)
  -tm [TM]    path to target mask. Type: nifti (.nii.gz)
```

### MSK Knee
```
usage: DOSMA knee [-h] [--ml] [--pid [PID]] [--fc] [--men] [--tc] [--t2]
                  [--t1_rho] [--t2_star]

optional arguments:
  -h, --help   show this help message and exit
  --ml         defines slices in sagittal direction going from medial ->
               lateral
  --pid [PID]  specify pid
  --fc         analyze femoral cartilage
  --men        analyze meniscus
  --tc         analyze tibial cartilage
  --t2         quantify t2
  --t1_rho     quantify t1_rho
  --t2_star    quantify t2_star
```

If no quantitative value flag (e.g. `--t2`, `--t1_rho`, `--t2_star`) is specified, all quantitative values will be calculated by default.

If no tissue flag (e.g. `--fc`) is specified, all tissues will be calculated by default.

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