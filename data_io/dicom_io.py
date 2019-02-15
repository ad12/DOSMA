import os

import numpy as np
import pydicom
import nibabel as nib
from natsort import natsorted

from data_io.format_io import DataReader, DataWriter
from data_io.med_volume import MedicalVolume
from data_io import orientation as stdo
import SimpleITK as sitk

__DICOM_EXTENSIONS__ = ('.dcm')
TOTAL_NUM_ECHOS_KEY = (0x19, 0x107e)


def contains_dicom_extension(a_str: str):
    bool_list = [a_str.endswith(ext) for ext in __DICOM_EXTENSIONS__]
    return bool(sum(bool_list))


def LPSplus_to_RASplus(headers):
    im_dir = headers[0].ImageOrientationPatient
    orientation = np.zeros([3, 3])
    i_vec, j_vec = im_dir[3:], im_dir[:3]  # unique to pydicom, please revise if using different library to load dicoms
    k_vec = np.asarray(headers[-1].ImagePositionPatient) - np.asarray(headers[0].ImagePositionPatient)
    k_vec = k_vec * np.dot(headers[0].ImagePositionPatient, np.cross(i_vec, j_vec))
    # k_div = np.abs(k_vec)
    # k_div[k_div == 0] += 1
    # orientation[:, 2] = k_vec / k_div
    orientation[:3, :3] = np.stack([i_vec, j_vec, k_vec], axis=1)
    scanner_origin = headers[0].ImagePositionPatient

    affine = np.zeros([4,4])
    affine[:3,:3] = orientation
    affine[:3, 3] = scanner_origin
    affine[:2,:] = -1*affine[:2,:]
    affine[3,3] = 1

    affine[affine == 0] = 0
    nib_axcodes = nib.aff2axcodes(affine)
    std_orientation = stdo.__orientation_nib_to_standard__(nib_axcodes)

    return std_orientation, affine[:3, 3]

class DicomReader(DataReader):
    """
    Key Notes:
    ------------------
    1. Dicom utilizes LPS convention:
        - LPS: right --> left, anterior --> posterior, inferior --> superior
        - we will call it LPS+, such that letters correspond to increasing end of axis
    """
    def load(self, dicom_dirpath):
        """Load dicoms into numpy array

        Required:
        :param dicom_dirpath: string path to directory where dicoms are stored

        Optional:
        :param dicom_ext: string extension for dicom files. Default is None, meaning the extension will not be checked

        :return tuple of MedicalVolume, list of dicom headers

        :raises NotADirectoryError if dicom pat does not exist or is not a directory

        Note: this function sorts files using natsort, an intelligent sorting tool.
            Please label your dicoms in a sequenced manner (e.g.: dicom1,dicom2,dicom3,...)
        """
        if not os.path.isdir(dicom_dirpath):
            raise NotADirectoryError("Directory %s does not exist" % dicom_dirpath)

        lstFilesDCM = []
        for dirName, subdirList, fileList in os.walk(dicom_dirpath):
            for filename in fileList:
                if contains_dicom_extension(filename):
                    lstFilesDCM.append(os.path.join(dirName, filename))

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No files found in directory %s" % dicom_dirpath)

        # Get reference file
        ref_dicom = pydicom.read_file(lstFilesDCM[0])
        max_num_echos = ref_dicom[TOTAL_NUM_ECHOS_KEY].value

        # Load spacing values (in mm)
        pixelSpacing = (float(ref_dicom.PixelSpacing[0]),
                        float(ref_dicom.PixelSpacing[1]),
                        float(ref_dicom.SpacingBetweenSlices))

        dicom_data = []
        for i in range(max_num_echos):
            dicom_data.append({'headers':[], 'arr':[]})

        for dicom_filename in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(dicom_filename, force=True)
            echo_number = ds.EchoNumbers
            dicom_data[echo_number-1]['headers'].append(ds)
            dicom_data[echo_number-1]['arr'].append(ds.pixel_array)

        vols = []
        for dd in dicom_data:
            headers = dd['headers']
            if len(headers) == 0:
                continue
            arr = np.stack(dd['arr'], axis=-1)

            rasplus_orientation, rasplus_origin = LPSplus_to_RASplus(headers)

            vol = MedicalVolume(arr,
                                pixel_spacing=pixelSpacing,
                                orientation=rasplus_orientation,
                                scanner_origin=rasplus_origin,
                                headers=headers)
            vols.append(vol)

        return vols

class DicomWriter(DataWriter):
    pass


if __name__ == '__main__':
    dicom_filepath = '../dicoms/healthy07/007'
    save_path = '../dicoms/healthy07/dcm-nifti.nii.gz'
    r = DicomReader()
    A = r.load(dicom_filepath)
    from data_io.nifti_io import NiftiWriter
    r = NiftiWriter()
    r.save(A[0], save_path)