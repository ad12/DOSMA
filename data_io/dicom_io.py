import os

import numpy as np
import pydicom
from natsort import natsorted

from data_io.format_io import DataReader, DataWriter
from data_io.med_volume import MedicalVolume


class DicomReader(DataReader):
    #
    # def __sitk_dicom_reader__(self, dicom_dirpath):
    #     reader = sitk.ImageSeriesReader()
    #     dicom_names = reader.GetGDCMSeriesFileNames(dicom_dirpath)
    #     reader.SetFileNames(dicom_names)
    #
    #     image = reader.Execute()

    def load_dicom(self, dicom_dirpath, dicom_ext=None):
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
                if (dicom_ext is None) or (dicom_ext is not None and dicom_ext in filename.lower()):
                    lstFilesDCM.append(os.path.join(dirName, filename))

        lstFilesDCM = natsorted(lstFilesDCM)
        if len(lstFilesDCM) == 0:
            raise FileNotFoundError("No files found in directory %s" % dicom_dirpath)

        # Get reference file
        ref_dicom = pydicom.read_file(lstFilesDCM[0])

        # Load dimensions based on rows, columns, and slices along Z axis
        pixelDims = (int(ref_dicom.Rows), int(ref_dicom.Columns), len(lstFilesDCM))

        # Load spacing values (in mm)
        pixelSpacing = (float(ref_dicom.PixelSpacing[0]),
                        float(ref_dicom.PixelSpacing[1]),
                        float(ref_dicom.SpacingBetweenSlices))

        dicom_array = np.zeros(pixelDims, dtype=ref_dicom.pixel_array.dtype)

        refs_dicom = []
        for dicom_filename in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(dicom_filename, force=True)
            refs_dicom.append(ds)
            dicom_array[:, :, lstFilesDCM.index(dicom_filename)] = ds.pixel_array

        volume = MedicalVolume(dicom_array, pixelSpacing)

        return volume, refs_dicom

class DicomWriter(DataWriter):
    pass