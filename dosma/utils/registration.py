import multiprocessing as mp
import os
import shutil
import uuid
from functools import partial
from typing import Sequence, Union

from nipype.interfaces.elastix import ApplyWarp, Registration
from nipype.interfaces.elastix.registration import RegistrationOutputSpec
from tqdm import tqdm

from dosma import file_constants as fc
from dosma.data_io.nifti_io import NiftiWriter, NiftiReader
from dosma.data_io.med_volume import MedicalVolume

MedVolOrPath = Union[MedicalVolume, str]


def register(
    target: MedVolOrPath,
    moving: Union[MedVolOrPath, Sequence[MedVolOrPath]],
    parameters: Union[str, Sequence[str]],
    output_path: str,
    target_mask: MedVolOrPath = None,
    moving_masks: Union[MedVolOrPath, Sequence[MedVolOrPath]] = None,
    num_workers: int = 0,
    num_threads: int = 1,
    show_pbar: bool = False,
    return_volumes: bool = False,
    **kwargs,
):
    """Register moving image(s) to the target.

    `MedVolOrPath` is a shorthand for `MedicalVolume` or `str`. It indicates the argument 
    can be either a `MedicalVolume` or a `str` path to a nifti file.

    Args:
        target (`MedicalVolume` or `str`): The target/fixed image.
        moving (`MedicalVolume`(s) or `str`(s)): The moving/source image(s).
        parameters (`str(s)`): Elastix parameter files to use.
        output_path (`str`): Output directory to store files.
        target_mask (`MedicalVolume` or `str`, optional): The target/fixed mask.
        moving_masks (`MedicalVolume`(s) or `str`(s), optional): The moving mask(s).
            If only one specified, the mask will be used for all moving images.
        num_workers (int, optional): Number of workers to use for reading/writing data and for registration.
            Note this is not used for registration, which is done via multiple threads - see `num_threads`
            for more details.
        num_threads (int, optional): Number of threads to use for registration. If `None`, defaults to 1.
        show_pbar (bool, optional): If `True`, show progress bar during registration. Note the progress bar
            will not be shown for intermediate reading/writing.
        kwargs: Keyword arguments used to initialize `nipype.interfaces.elastix.Registration`

    Returns:
        
    """
    has_output_path = bool(output_path)
    if not output_path:
        output_path = os.path.join(fc.TEMP_FOLDER_PATH, "register")

    moving = [moving] if isinstance(moving, (MedicalVolume, str)) else moving
    moving_masks = [moving_masks] if moving_masks is None or isinstance(moving_masks, (MedicalVolume, str)) else moving_masks
    if len(moving_masks) > 1 and len(moving) != len(moving_masks):
        raise ValueError("Got {} moving images but {} moving masks".format(len(moving), len(moving_masks)))

    files = [target, target_mask] + moving + moving_masks

    # Write medical volumes (if any) to nifti file for use with elastix.
    tmp_dir = os.path.join(output_path, "tmp")
    default_files = ["target", "target-mask"] + [f"moving-{idx}" for idx in range(len(moving))] + [f"moving-mask-{idx}" for idx in range(len(moving_masks))]  #noqa
    assert len(default_files) == len(files), default_files  # should be 1-to-1 with # args provided
    vols = [(idx, v) for idx, v in enumerate(files) if isinstance(v, MedicalVolume)]
    idxs, vols = [x[0] for x in vols], [x[1] for x in vols]
    if len(vols) > 0:
        filepaths = [os.path.join(tmp_dir, f"{default_files[idx]}.nii.gz") for idx in idxs]
        if num_workers > 0:
            with mp.Pool(min(num_workers, len(vols))) as p:
                out = p.starmap_async(_write, zip(vols, filepaths))
                out.wait()
        else:
            for vol, fp in zip(vols, filepaths):
                _write(vol, fp)
        for idx, fp in zip(idxs, filepaths):
            files[idx] = fp
    
    # Assign file paths to respective variables.
    target, moving = files[0], files[2:2+len(moving)]
    target_mask, moving_masks = files[1], files[2+len(moving):]
    if len(moving_masks) == 1:
        moving_masks = moving_masks * len(moving)
    
    # Perform registration.
    out = []
    for idx, (mvg, mvg_mask) in tqdm(
        enumerate(zip(moving, moving_masks)), disable=not show_pbar, total=len(moving)
    ):
        out_path = os.path.join(output_path, f"moving-{idx}")
        _out = _elastix_register(
            target, mvg, parameters, out_path, target_mask, 
            mvg_mask, False, num_threads, **kwargs,
        )
        out.append(_out)
    out = tuple(out)

    # Load volumes.
    if return_volumes:
        filepaths = [x[-1].warped_file if isinstance(x, Sequence) else x.warped_file for x in out]
        if num_workers > 0:
            with mp.Pool(min(num_workers, len(filepaths))) as p:
                vols = p.map(_read, filepaths)
        else:
            for fp in filepaths:
                vols = _read(fp)
        out = out, tuple(vols)
    
    # Clean up.
    for _dir in [tmp_dir, output_path if not has_output_path else None]:
        if not _dir or not os.path.isdir(_dir):
            continue
        shutil.rmtree(_dir)

    return out


def apply_warp(
    moving: MedVolOrPath,
    transform: Union[str, Sequence[str]] = None,
    out_registration: RegistrationOutputSpec = None,
    output_path: str = None,
    rtype: type = MedicalVolume,
    num_threads: int = 1,
    show_pbar: bool = False,
) -> MedVolOrPath:
    """Apply transform(s) to moving image using transformix.

    Use transformix to apply a transform on an input image. The transform(s) is/are
    specified in the transform-parameter file(s).

    Args:
        moving (MedicalVolume(s) or str(s)): The moving/source image to transform.
        transform (str(s)): Paths to transform files to be used by transformix.
            If multiple files provided, transforms will be applied sequentially.
            If `None`, will be determined by `out_registration.transform`.
        out_registration (RegistrationOutputSpec(s)): Outputs from elastix registration
            using nipype. Must be specified if `transform` is None.
        output_path (str): Output directory to store files.
        rtype (type, optional): Return type - either `MedicalVolume` or `str`.
            If `str`, `output_path` must be specified. Defaults to `MedicalVolume`.
        num_threads (int, optional): Number of threads to use for registration. If `None`, defaults to 1.
        show_pbar (bool, optional): If `True`, show progress bar when applying transforms. 

    Return:
        MedVolOrPath: The medical volume or nifti file corresponding to the volume.
            See `rtype` for details.
    """
    assert rtype in [MedicalVolume, str], rtype  # rtype must be MedicalVolume or str
    has_output_path = bool(output_path)
    if rtype == str and not has_output_path:
        raise ValueError("`output_path` must be specified when `rtype=str`")
    if not output_path:
        output_path = os.path.join(fc.TEMP_FOLDER_PATH, f"apply_warp-{str(uuid.uuid1())}")
    os.makedirs(output_path, exist_ok=True)

    if not transform:
        transform = out_registration.transform
    elif not isinstance(transform, Sequence):
        transform = [transform]

    mv_filepath = os.path.join(output_path, "moving.nii.gz")
    if isinstance(moving, MedicalVolume):
        NiftiWriter().save(moving, mv_filepath)
        moving = mv_filepath
    
    for tf in tqdm(transform, disable=not show_pbar):
        reg = ApplyWarp()
        reg.inputs.moving_image = moving
        reg.inputs.transform_file = tf
        reg.inputs.output_path = output_path
        reg.terminal_output = fc.NIPYPE_LOGGING
        reg.inputs.num_threads = num_threads
        reg_output = reg.run()

        moving = reg_output.outputs.warped_file

    if rtype == MedicalVolume:
        out = NiftiReader().load(moving)
    else:
        out = moving

    if os.path.isfile(mv_filepath):
        os.remove(mv_filepath)
    if not has_output_path:
        shutil.rmtree(output_path)

    return out


def _elastix_register(
    target: str, moving: str, parameters: Sequence[str], output_path: str,
    target_mask: str=None, moving_mask: str=None, sequential=False, num_threads=None, 
    **kwargs,
):
    if sequential:
        raise ValueError("`sequential` not yet supported")

    os.makedirs(output_path, exist_ok=True)

    reg = Registration()
    reg.inputs.fixed_image = target
    reg.inputs.moving_image = moving
    if isinstance(parameters, str):
        parameters = [parameters]
    reg.inputs.output_path = output_path
    reg.terminal_output = fc.NIPYPE_LOGGING
    if num_threads:
        reg.inputs.num_threads = num_threads
    if target_mask:
        reg.inputs.fixed_mask = target_mask
    if moving_mask:
        reg.inputs.target_mask = moving_mask
    reg.inputs.parameters = parameters

    for k, v in kwargs.items():
        setattr(reg.inputs, k, v)

    return reg.run().outputs


def _write(vol: MedicalVolume, path: str):
    """Extracted out for multiprocessing purposes."""
    NiftiWriter().save(vol, path)


def _read(path: str):
    return NiftiReader().load(path)
