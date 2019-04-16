import os, sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
import time

# Matplotlib initialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.insert(0, '../')
from data_io.dicom_io import DicomReader
from data_io.orientation import SAGITTAL

def normalize_im(x):
    return (x - np.min(x)) / np.max(x)

def write_subplot(x, filepath):
    print('Saving %s' % filepath)
    x = np.squeeze(x)
    x = normalize_im(x)

    nrows = 5
    ncols = 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
    count = 0
    for i in np.linspace(0, x.shape[0]-1, nrows*ncols):
        slice_ind = int(i)
        slice_title = 'Slice %d' % (slice_ind + 1)
        ax = axs[int(count / ncols)][count % ncols]
        ax.imshow(x[slice_ind, ...], cmap='gray')
        ax.set_title(slice_title)
        ax.axis('off')
        count += 1
    plt.savefig(filepath)
    plt.close()

if __name__ == '__main__':
    dirpath = '/bmrNAS/people/akshay/dicoms/DESS_SR_Output/'
    save_path = '/bmrNAS/people/akshay/dicoms/DESS_SR_Output/tiles'
    dr = DicomReader()

    for pid in range(1,52):
        pid_str = '%02d' % pid
        dicom_path = os.path.join(dirpath, pid_str, 'DESS')
        dcm_vol = dr.load(dicom_path)
        dcm_vol = dcm_vol[0]
        dcm_vol.reformat(SAGITTAL)

        save_fp = os.path.join(save_path, '%s.png' % pid_str)
