import os

import nibabel as nib
import nibabel.orientations as nibo
import numpy as np

from data_io.format_io import DataReader, DataWriter
from data_io.med_volume import MedicalVolume
from data_io.orientation import __orientation_nib_to_standard__, __orientation_standard_to_nib__
from utils import io_utils

__NIFTI_EXTENSIONS__ = ('.nii', '.nii.gz')


def contains_nifti_extension(a_str: str):
    bool_list = [a_str.endswith(ext) for ext in __NIFTI_EXTENSIONS__]
    return bool(sum(bool_list))


class NiftiReader(DataReader):
    def __get_pixel_spacing__(self, nib_affine):
        col_i, col_j, col_k = nib_affine[..., 0], nib_affine[..., 1], nib_affine[..., 2]

        ps_i = col_i[np.nonzero(col_i)]
        ps_j = col_j[np.nonzero(col_j)]
        ps_k = col_k[np.nonzero(col_k)]

        assert len(ps_i) == 1 and len(ps_j) == 1 and len(ps_k) == 1, \
            "Multiple nonzero values found: There should only be 1 nonzero element in first 3 columns of Nibabel affine matrix"

        return (abs(ps_i[0]), abs(ps_j[0]), abs(ps_k[0]))

    def load(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError('%s not found' % filepath)

        if not contains_nifti_extension(filepath):
            raise ValueError('%s must be a file with extension `.nii` or `.nii.gz`' % filepath)

        nib_img = nib.load(filepath)
        nib_img_affine = nib_img.affine
        orientation = __orientation_nib_to_standard__(nib.aff2axcodes(nib_img_affine))
        origin = tuple(nib_img_affine[:3, 3])
        pixel_spacing = self.__get_pixel_spacing__(nib_img_affine)
        np_img = nib_img.get_fdata()

        return MedicalVolume(np_img, pixel_spacing, orientation, origin)


class NiftiWriter(DataWriter):
    def __get_nib_affine__(self, im: MedicalVolume):
        pixel_spacing = im.pixel_spacing
        origin = im.scanner_origin

        nib_orientation_inds = nibo.axcodes2ornt(__orientation_standard_to_nib__(im.orientation))
        assert nib_orientation_inds.shape == (3, 2), "Currently only supporting perfectly orthogonal scans"

        nib_affine = np.zeros([4, 4])
        rows, _ = nib_orientation_inds.shape
        for r in range(rows):
            ind, val = nib_orientation_inds[r, 0], nib_orientation_inds[r, 1]
            nib_affine[int(ind), r] = val * pixel_spacing[r]

        nib_affine[:, 3] = (np.append(np.asarray(origin), 1)).flatten()

        return nib_affine

    def save(self, im: MedicalVolume, filepath: str):
        if not contains_nifti_extension(filepath):
            raise ValueError('%s must be a file with extension `.nii` or `.nii.gz`' % filepath)

        # Create dir if does not exist
        io_utils.check_dir(os.path.dirname(filepath))

        nib_affine = self.__get_nib_affine__(im)
        np_im = im.volume
        nib_img = nib.Nifti1Image(np_im, nib_affine)

        nib.save(nib_img, filepath)


if __name__ == '__main__':
    load_filepath = '../dicoms/healthy07/h7.nii.gz'
    save_filepath = '../dicoms/healthy07/h7_nifti_writer-transpose.nii.gz'
    save_filepath2 = '../dicoms/healthy07/h7_nifti_writer-flip.nii.gz'

    nr = NiftiReader()
    med_vol = nr.load(load_filepath)
    o = med_vol.orientation

    nw = NiftiWriter()
    o1 = (o[1], o[0], o[2])
    med_vol.reformat(o1)
    nw.save(med_vol, save_filepath)

    o2 = (o[0][::-1], o[1], o[2])
    med_vol.reformat(o2)
    nw.save(med_vol, save_filepath2)

    print('')
    a = nr.load(load_filepath)
    b = nr.load(save_filepath)
    c = nr.load(save_filepath2)
