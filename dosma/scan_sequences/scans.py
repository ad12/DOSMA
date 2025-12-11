"""Abstract classes interface implementation of scan.

This module defines the abstract interfaces that can be extended by concrete scan types.

Different scan types produce images with different qualities
(different quantitative parameters, resolutions, etc.).
These scan types have different actions (or processing) associated with them.

A scan can have multiple volumes if multiple phases or echo times are used to image in the scan.

Scans that have high resolution & signal-to-noise ratio (SNR) are typically used for
segmentation. Because of this property, they are often referred to as *targets*.
This allows them to serve as a good template that other lower SNR/resolution scans
can be registered to.
"""

import logging
import os
import re
from abc import abstractmethod
from time import localtime, strftime
from typing import Any, Sequence, Union

import numpy as np
import scipy.ndimage as sni
from natsort import natsorted

from dosma.core.io import format_io_utils as fio_utils
from dosma.core.io.nifti_io import NiftiReader
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.scan_sequences.scan_io import ScanIOMixin
from dosma.tissues.tissue import Tissue
from dosma.utils import env, io_utils

__all__ = ["ScanSequence"]

_logger = logging.getLogger(__name__)


class ScanSequence(ScanIOMixin):
    """The class for scan sequences and corresponding analysis.

    This is the base class for scan-specific analysis. All classes implementing scan-specific
    image processing/analysis methods should inherit from this class.

    Args:
        volumes (MedicalVolume(s)): The medical image(s) that this scan is composed of.
            Note these are typically the different 2D/3D image(s) . For example, for a
            multi-echo MRI scan, each volume in ``volumes`` would correspond to the spatial
            volume acquired at different echo times.

    Attributes:
        volumes (MedicalVolume(s)): See ``volume`` in Args.
        temp_path (str): The directory path where temporary results
            are written. This will be under ``dosma.file_constants.TEMP_FOLDER_PATH``.

    Raises:
        NotADirectoryError: If `dicom_path` is not a valid directory.
        ValueError: If dicoms do not correspond to the expected sequence.
    """

    NAME = ""
    __DEFAULT_SPLIT_BY__ = "EchoNumbers"

    def __init__(self, volumes: Union[MedicalVolume, Sequence[MedicalVolume]]):
        self.volumes = volumes

        # TODO: Remove series number as an attribute.
        # It should be directly accessed through the reference header.
        # This involves some backward compatible fixes.
        self.series_number = None
        self._from_file_args = {}

        self.temp_path = os.path.join(
            env.temp_dir(), self.NAME, strftime("%Y-%m-%d-%H-%M-%S", localtime())
        )
        self.tissues = []
        self._metadata = {}

    def __validate_scan__(self) -> bool:
        """Validate this scan (usually done by checking dicom header tags, if available).
        Returns:
            bool: `True` if scan metadata is valid, `False` otherwise.
        """
        return True

    def get_metadata(self, key: Any, default=None):
        """Get metadata for the scan.

        The metadata is either stored in the ``self._metadata`` property
        if it is not a part of the standard DICOM header or the reference
        DICOM does not exist.

        Args:
            key (Any): The attribute to fetch.
            default (Any, optional): The default value to return.
                Set to ``False`` to raise an error when metadata is not found in
                either the scan metadata or DICOM headers (if available).
                Defaults to ``None``.

        Returns:
            Any: The value.

        Raises:
            ValueError: If ``default == False`` and metadata not found
        """
        metadata = self._metadata.get(key, None)
        if metadata is None and self.ref_dicom is not None:
            metadata = self.ref_dicom[key].value if key in self.ref_dicom else None
        if metadata is None and default is False:
            raise KeyError(f"Metadata '{key}' not found")
        elif metadata is None:
            return default
        else:
            return metadata

    def get_dimensions(self):
        """Get shape of volumes.

        All volumes in scan are assumed to be same dimension.

        Returns:
            tuple[int]: Shape of each volume in scan.
        """
        if isinstance(self.volumes, MedicalVolume):
            return self.volumes.shape
        return self.volumes[0].shape

    @property
    def ref_dicom(self):
        """The reference dicom.

        The reference dicom is defined as the first dicom of the first volume if
        ``self.volumes`` is a sequence of volumes.
        """
        vol = self.volumes[0] if isinstance(self.volumes, Sequence) else self.volumes
        headers = vol.headers(flatten=True)
        return headers[0] if headers is not None else None

    def __add_tissue__(self, new_tissue: Tissue):
        """Add a tissue to the list of tissues associated with this scan.

        Args:
            new_tissue (Tissue): Tissue to add.

        Raises:
            ValueError: If tissue already exists in list.
                For example, we cannot add FemoralCartilage twice to the list of tissues.
        """
        contains_tissue = any(tissue.ID == new_tissue.ID for tissue in self.tissues)
        if contains_tissue:
            raise ValueError("Tissue already exists")

        self.tissues.append(new_tissue)

    def to(self, device):
        """Moves volumes of this scan onto the appropriate device.

        This is an inplace operation. A new ScanSequence will not
        be returned.

        Args:
            device: The device to move to.

        Returns:
            self
        """
        if isinstance(self.volumes, MedicalVolume):
            self.volumes = self.volumes.to(device)
            return

        self.volumes = [v.to(device) for v in self.volumes]

        return self


class NonTargetSequence(ScanSequence):
    """
    Abstract class for scans that cannot serve as targets and have
    to be registered to some other target scan.

    Examples: Cubequant, Cones
    """

    @abstractmethod
    def interregister(self, target_path: str, mask_path: str = None):
        """Register scan to the target scan - save as parameter in scan (volumes, subvolumes, etc).

        We use the term *interregister* to refer to registration between volumes of different scans.
        Conversely, *intraregister* refers to registering volumes from the same scan.

        If there are many subvolumes to interregister with the base scan,
        typically the following actions are used to reduce error accumulation
        with multiple registrations:

        1. Pick the volume with the highest SNR. Call this the base moving image.
        2. Register the base moving image to the target image using elastix.
        3. Capture the transformation file(s) detailing the transforms to go from
           base moving image -> target image.
        4. Apply these transformation(s) to the remaining volumes.

        Args:
            target_path (str): Path to NIfTI file storing scan.
                This scan will serve as the base for registration.
                Note the best target scan will typically have a high SNR.

            mask_path (str): Path to mask to use to use as focus points for registration.
        """
        pass  # pragma: no cover

    def __load_interregistered_files__(self, interregistered_dirpath: str):
        """Load the NIfTI files of the interregistered subvolumes.

        These subvolumes have already been registered to some target scan
        using the `interregister` function.

        Args:
            interregistered_dirpath (str): Directory path where interregistered volumes are stored.

        Returns:
            Tuple[Dict[int, MedicalVolume], List[float]]:
                A dictionary mapping echo time index -> MedicalVolume

        Raises:
            ValueError: If files are not of the name `<INTEGER>.nii.gz`
                (e.g. `0.nii.gz`, `000.nii.gz`, etc.) or if no interregistered files
                found in interregistered_dirpath.
        """
        _logger.info("Loading interregistered files")
        if "interregistered" not in interregistered_dirpath:
            raise ValueError("Invalid path for loading {} interregistered files".format(self.NAME))

        subfiles = os.listdir(interregistered_dirpath)
        subfiles = natsorted(subfiles)

        if len(subfiles) == 0:
            raise ValueError("No interregistered files found")

        indices = []
        subvolumes = []
        nifti_reader = NiftiReader()
        for subfile in subfiles:
            subfile_nums = re.findall(r"[-+]?\d*\.\d+|\d+", subfile)
            if len(subfile_nums) == 0:
                raise ValueError("{} is not an interregistered '.gz.nii' file.".format(subfile))

            subfile_num = int(subfile_nums[0])
            indices.append(subfile_num)

            filepath = os.path.join(interregistered_dirpath, subfile)
            subvolume = nifti_reader.load(filepath)

            subvolumes.append(subvolume)

        assert len(indices) == len(subvolumes), "Number of subvolumes mismatch"

        if len(subvolumes) == 0:
            raise ValueError("No interregistered files found")

        subvolumes_dict = {}
        for i in range(len(indices)):
            subvolumes_dict[indices[i]] = subvolumes[i]

        return subvolumes_dict

    def __dilate_mask__(
        self,
        mask_path: str,
        temp_path: str,
        dil_rate: float = preferences.mask_dilation_rate,
        dil_threshold: float = preferences.mask_dilation_threshold,
    ):
        """Dilate mask using gaussian blur and write to disk to use with Elastix.

        Args:
            mask_path (str | MedicalVolume): File path for mask or mask to use to use as
                focus points for registration. Mask must be binary.
            temp_path (str): Directory path to store temporary data.
            dil_rate (`float`, optional): Dilation rate (sigma).
                Defaults to ``preferences.mask_dilation_rate``.
            dil_threshold (`float`, optional): Threshold to binarize dilated mask.
                Must be between [0, 1]. Defaults to ``preferences.mask_dilation_threshold``.

        Returns:
            str: File path of dilated mask.

        Raises:
            FileNotFoundError: If `mask_path` not valid file.
            ValueError: If `dil_threshold` not in range [0, 1].
        """

        if dil_threshold < 0 or dil_threshold > 1:
            raise ValueError("'dil_threshold' must be in range [0, 1]")

        if isinstance(mask_path, MedicalVolume):
            mask = mask_path
        elif os.path.isfile(mask_path):
            mask = fio_utils.generic_load(mask_path, expected_num_volumes=1)
        else:
            raise FileNotFoundError("File {} not found".format(mask_path))

        dilated_mask = (
            sni.gaussian_filter(np.asarray(mask.volume, dtype=np.float32), sigma=dil_rate)
            > dil_threshold
        )
        fixed_mask = np.asarray(dilated_mask, dtype=np.int8)
        fixed_mask_filepath = os.path.join(io_utils.mkdirs(temp_path), "dilated-mask.nii.gz")

        dilated_mask_volume = MedicalVolume(fixed_mask, affine=mask.affine)
        dilated_mask_volume.save_volume(fixed_mask_filepath)

        return fixed_mask_filepath
