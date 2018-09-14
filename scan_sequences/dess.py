from scan_sequences.scans import TargetSequence
from utils import dicom_utils
from utils import im_utils
import os

class Dess(TargetSequence):
    NUM_ECHOS = 2
    subvolumes = None

    def segment(self, model, tissue):
        if self.subvolumes is None:
            self.subvolumes = dicom_utils.split_volume(self.volume, echos=self.NUM_ECHOS)

        # Use first echo for segmentation
        segmentation_volume = self.subvolumes[0]
        volume = dicom_utils.whiten_volume(segmentation_volume)

        # Segment tissue and add it to list
        mask = model.generate_mask(volume)
        tissue.mask = mask
        self.__add_tissue__(tissue)

        return mask

    def save_tissue_masks(self, dirpath, ext='tiff'):
        for tissue in self.tissues:
            filepath = os.path.join(dirpath, '%s.%s' % (tissue.NAME, ext))
            im_utils.write_3d(filepath, tissue.mask)