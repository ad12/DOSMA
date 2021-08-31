import itertools
import logging
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import uuid
import warnings
from functools import partial
from typing import Dict, Sequence, Union

from nipype.interfaces.elastix import ApplyWarp, Registration
from nipype.interfaces.elastix.registration import RegistrationOutputSpec
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dosma import file_constants as fc
from dosma.core.device import cpu_device
from dosma.core.io.nifti_io import NiftiReader, NiftiWriter
from dosma.core.med_volume import MedicalVolume
from dosma.defaults import preferences
from dosma.utils import env

__all__ = ["register", "apply_warp", "symlink_elastix", "unlink_elastix"]

MedVolOrPath = Union[MedicalVolume, str]
_logger = logging.getLogger(__name__)


def register(
    target: MedVolOrPath,
    moving: Union[MedVolOrPath, Sequence[MedVolOrPath]],
    parameters: Union[str, Sequence[str]],
    output_path: str,
    target_mask: MedVolOrPath = None,
    moving_masks: Union[MedVolOrPath, Sequence[MedVolOrPath]] = None,
    sequential: bool = False,
    collate: bool = True,
    num_workers: int = 0,
    num_threads: int = 1,
    show_pbar: bool = False,
    return_volumes: bool = False,
    rtype: type = dict,
    **kwargs,
):
    """Register moving image(s) to the target.

    ``MedVolOrPath`` is a shorthand for ``Union[MedicalVolume, str, Path]``.
    It indicates the argument can be either a ``MedicalVolume`` or a path to a nifti file.

    Args:
        target (`MedicalVolume` or `str`): The target/fixed image.
        moving (MedicalVolume(s) or str(s)): The moving/source image(s).
        parameters (str(s)): Elastix parameter files to use.
        output_path (str): Output directory to store files.
        target_mask (MedicalVolume or str, optional): The target/fixed mask.
        moving_masks (MedicalVolume(s) or str(s), optional): The moving mask(s).
            If only one specified, the mask will be used for all moving images.
        sequential (bool, optional): If `True`, apply parameter files sequentially.
        collate (bool, optional): If `True`, will collate outputs from sequential registration
            into single RegistrationOutputSpec instance. If `sequential=False`, this argument
            is ignored.
        num_workers (int, optional): Number of workers to use for reading and writing data and
            for registration.
        num_threads (int, optional): Number of threads to use for registration.
            Note total number of threads used will be ``num_workers * num_threads``.
        show_pbar (bool, optional): If `True`, show progress bar during registration.
            Note the progress bar will not be shown for intermediate reading/writing.
        return_volumes (bool, optional): If `True`, registered volumes will also be returned.
            By default, only the output namespaces (RegistrationOutputSpec) of the registrations are
            returned.
        rtype (type, optional): The return type. Either `dict` or `tuple`.
        kwargs: Keyword arguments used to initialize `nipype.interfaces.elastix.Registration`.

    Returns:
        Dict or Tuple:
            Type specified by ``rtype``. If ``rtype=dict``, returns dict with keys ``'outputs'``
            and ``'volumes'`` (if ``return_volumes=True``). If ``rtype=tuple``, returns
            ``(outputs, volumes or None)``. Length of ``outputs`` and ``volumes`` depends on
            number of images specified in ``moving``:

            outputs (Sequence[RegistrationOutputSpec]): The output objects from
            elastix registration, one for each moving image. Each object is effectively
            a namespace with four main attributes:

                - 'transform' (List[str]): Paths to transform files produced using registration.
                - 'warped_file' (str): Path to the final registered image.
                - 'warped_files' (List[str]): Paths to all intermediate images created
                  if multiple parameter files used.

            volumes (Sequence[MedicalVolume]): Registered volumes.
    """
    assert issubclass(rtype, (Dict, Sequence))  # `rtype` must be dict or tuple
    has_output_path = bool(output_path)
    if not output_path:
        output_path = os.path.join(
            env.temp_dir(), f"register-{str(uuid.uuid1())}-{str(uuid.uuid4())}"
        )

    moving = [moving] if isinstance(moving, (MedicalVolume, str)) else moving
    moving_masks = (
        [moving_masks]
        if moving_masks is None or isinstance(moving_masks, (MedicalVolume, str))
        else moving_masks
    )
    if len(moving_masks) > 1 and len(moving) != len(moving_masks):
        raise ValueError(
            "Got {} moving images but {} moving masks".format(len(moving), len(moving_masks))
        )

    files = [target, target_mask] + moving + moving_masks
    if any(isinstance(f, MedicalVolume) and f.device != cpu_device for f in files):
        raise RuntimeError("MedicalVolume data must be on CPU")

    # Write medical volumes (if any) to nifti file for use with elastix.
    tmp_dir = os.path.join(output_path, "tmp")
    default_files = (
        ["target", "target-mask"]
        + [f"moving-{idx}" for idx in range(len(moving))]
        + [f"moving-mask-{idx}" for idx in range(len(moving_masks))]
    )  # noqa
    assert len(default_files) == len(files), default_files  # should be 1-to-1 with # args provided
    vols = [(idx, v) for idx, v in enumerate(files) if isinstance(v, MedicalVolume)]
    idxs, vols = [x[0] for x in vols], [x[1] for x in vols]

    # Temporary directory must be created prior to writing data
    # due to issues with creating directories in multiprocessing settings.
    os.makedirs(tmp_dir, exist_ok=True)

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
    target, moving = files[0], files[2 : 2 + len(moving)]
    target_mask, moving_masks = files[1], files[2 + len(moving) :]
    if len(moving_masks) == 1:
        moving_masks = moving_masks * len(moving)

    all_outputs = {}

    # Perform registration.
    reg_out_paths = [os.path.join(output_path, f"moving-{idx}") for idx in range(len(moving))]
    reg_args = list(zip(moving, moving_masks, reg_out_paths))
    if num_workers > 0:
        func = partial(
            _elastix_register_mp,
            target=target,
            parameters=parameters,
            target_mask=target_mask,
            sequential=sequential,
            collate=collate,
            num_threads=num_threads,
            **kwargs,
        )
        max_workers = min(num_workers, len(reg_args))
        out = process_map(
            func, reg_args, max_workers=max_workers, tqdm_class=tqdm, disable=not show_pbar
        )
    else:
        out = []
        for mvg, mvg_mask, out_path in tqdm(reg_args, disable=not show_pbar):
            _out = _elastix_register(
                target,
                mvg,
                parameters,
                out_path,
                target_mask,
                mvg_mask,
                sequential,
                collate,
                num_threads,
                **kwargs,
            )
            out.append(_out)

    all_outputs["outputs"] = tuple(out)

    # Load volumes.
    if return_volumes:
        filepaths = [x[-1].warped_file if isinstance(x, Sequence) else x.warped_file for x in out]
        if num_workers > 0:
            with mp.Pool(min(num_workers, len(filepaths))) as p:
                vols = p.map(_read, filepaths)
        else:
            vols = []
            for fp in filepaths:
                vols.append(_read(fp))
        all_outputs["volume"] = tuple(vols)

    # Clean up.
    for _dir in [tmp_dir, output_path if not has_output_path else None]:
        if not _dir or not os.path.isdir(_dir):
            continue
        shutil.rmtree(_dir)

    if issubclass(rtype, dict):
        out = rtype(all_outputs)
    elif issubclass(rtype, Sequence):
        out = rtype([all_outputs["outputs"], all_outputs.get("volume", None)])
    else:
        assert False  # Should have type checking earlier.

    return out


def apply_warp(
    moving: Union[MedVolOrPath, Sequence[MedVolOrPath]],
    transform: Union[str, Sequence[str]] = None,
    out_registration: RegistrationOutputSpec = None,
    output_path: Union[str, Sequence[str]] = None,
    rtype: type = MedicalVolume,
    num_threads: int = 1,
    show_pbar: bool = False,
    num_workers: int = 0,
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
        num_threads (int, optional): Number of threads to use for registration.
            If `None`, defaults to 1.
        show_pbar (bool, optional): If `True`, show progress bar when applying transforms.

    Return:
        MedVolOrPath: The medical volume or nifti file corresponding to the volume.
            See `rtype` for details.
    """
    single_vol = isinstance(moving, (MedicalVolume, os.PathLike))
    if single_vol:
        if num_workers > 0:
            _logger.warning("Ignoring `num_workers` - only single volume was detected")
        return _apply_warp(
            moving=moving,
            transform=transform,
            out_registration=out_registration,
            output_path=output_path,
            rtype=rtype,
            num_threads=num_threads,
            show_pbar=show_pbar,
        )

    num_volumes = len(moving)
    seq_type = type(moving)

    if not output_path:
        output_path = [None] * num_volumes
    elif isinstance(output_path, (str, os.PathLike)):
        output_path = [os.path.join(output_path, f"image-{idx}") for idx in range(num_volumes)]
    elif not isinstance(output_path, Sequence) or len(output_path) != num_volumes:
        raise ValueError(
            "`output_path` must be a directory or list of directories " "of same length as `moving`"
        )

    warp_args = list(zip(moving, output_path))
    if num_workers > 0:
        func = partial(
            _apply_warp_mp,
            transform=transform,
            out_registration=out_registration,
            rtype=rtype,
            num_threads=num_threads,
            show_pbar=False,
        )
        max_workers = min(num_workers, len(warp_args))
        out = process_map(
            func, warp_args, max_workers=max_workers, tqdm_class=tqdm, disable=not show_pbar
        )
    else:
        out = []
        for mvg, out_path in tqdm(warp_args, disable=not show_pbar):
            _out = _apply_warp(
                moving=mvg,
                output_path=out_path,
                transform=transform,
                out_registration=out_registration,
                rtype=rtype,
                num_threads=num_threads,
                show_pbar=False,
            )
            out.append(_out)

    return seq_type(out)


def symlink_elastix(path: str = None, lib_only: bool = True, force: bool = False):
    """Symlinks elastix/transformix files to the dosma library.

    Args:
        path (str, optional): Path to elastix folder. This folder should
            contain two folders `bin` and `lib`. If `None`, determined
            using `which elastix`. This will overwrite existing linked
            files. path cannot be automatically determined on Windows.
        lib_only (bool, optional): If `True`, only links contents of `lib`
            folder.
        force (bool, optional): If `True`, unlinks existing files before relinking.
            Note this operation is not atomic.

    Note:
        Setting elastix paths this way is not recommended unless you
        are using a MacOS (Darwin) platform, where there are known
        path issues with elastix (https://github.com/almarklein/pyelastix/issues/9).
        For linux and windows machines, using the setup described in the elastix
        guide is sufficient.
    """
    system = platform.system().lower()
    assert system in ["windows", "darwin", "linux"]
    if system != "darwin":
        warnings.warn(
            f"Symlinking elastix/transformix paths not recommended for {system} " f"machines"
        )

    if path is None:
        if system == "windows":
            raise ValueError("`path` cannot be determined automatically on Windows")
        try:
            out = subprocess.check_output(["which", "elastix"]).decode("ascii").strip("\n")
            path = os.path.dirname(os.path.dirname(out))
        except subprocess.CalledProcessError:
            raise ValueError(
                "Path to `elastix` not intialized. "
                "Use `export PATH=/path/to/elastix/folder:$PATH`"
            )
    assert os.path.isdir(path), path  # must be a directory

    dirs = {"lib": [x for x in os.listdir(os.path.join(path, "lib")) if x.startswith("libANNlib")]}
    if not lib_only:
        dirs["bin"] = ["elastix", "transformix"]

    for dirname, files in dirs.items():
        for file in files:
            src = os.path.join(path, dirname, file)
            tgt = os.path.join(fc._DOSMA_ELASTIX_FOLDER, file)
            if os.path.exists(tgt):
                if force:
                    os.remove(tgt)
                else:
                    raise FileExistsError(
                        f"File {tgt} exists. "
                        f"Use `unlink_elastix` or `force` to unlink the file."
                    )
            os.symlink(src, tgt)


def unlink_elastix():
    """Unlinks all elastix/transformix files in the dosma library."""
    for x in os.listdir(fc._DOSMA_ELASTIX_FOLDER):
        x = os.path.join(fc._DOSMA_ELASTIX_FOLDER, x)
        if os.path.islink(x):
            os.remove(x)


def _elastix_register(
    target: str,
    moving: str,
    parameters: Sequence[str],
    output_path: str,
    target_mask: str = None,
    moving_mask: str = None,
    sequential=False,
    collate=True,
    num_threads=None,
    use_mask: Sequence[bool] = None,
    **kwargs,
):
    def _register(_moving, _parameters, _output_path, _use_mask=None):
        if isinstance(_parameters, str):
            _parameters = [_parameters]
        if _use_mask is None:
            _use_mask = target_mask is not None or moving_mask is not None

        _output_path = os.path.abspath(_output_path)
        os.makedirs(_output_path, exist_ok=True)

        elastix_path = _local_exe("elastix")
        cwd = _local_lib_dir()

        reg = Registration()
        if elastix_path:
            reg._cmd = elastix_path
        reg.inputs.fixed_image = os.path.abspath(target)
        reg.inputs.moving_image = os.path.abspath(_moving)
        reg.inputs.parameters = [os.path.abspath(p) for p in _parameters]
        reg.inputs.output_path = os.path.abspath(_output_path)
        reg.terminal_output = preferences.nipype_logging
        if num_threads:
            reg.inputs.num_threads = num_threads
        if _use_mask and target_mask is not None:
            reg.inputs.fixed_mask = os.path.abspath(target_mask)
        if _use_mask and moving_mask is not None:
            reg.inputs.target_mask = os.path.abspath(moving_mask)
        for k, v in kwargs.items():
            setattr(reg.inputs, k, v)

        return reg.run(cwd=cwd).outputs

    def _collate_outputs(_outs):
        """
        Concatenates fields that are sequential and takes final output
        for fields that are not.
        """
        if len(_outs) == 1:
            return _outs[0]

        _result = _outs[0]
        fields = list(_outs[0].__dict__.keys())
        for _fld in fields:
            _res_val = getattr(_result, _fld)
            if not isinstance(_res_val, str) and isinstance(_res_val, Sequence):
                val = list(itertools.chain.from_iterable([getattr(x, _fld) for x in _outs]))
            else:
                val = getattr(_outs[-1], _fld)
            setattr(_result, _fld, val)
        return _result

    if use_mask is not None:
        assert sequential  # use_mask can only be specified when sequential is specified
    if sequential:
        outs, mvg = [], moving
        for idx, param in enumerate(parameters):
            _use_mask = None if use_mask is None else use_mask[idx]
            _out = _register(mvg, param, os.path.join(output_path, f"param{idx}"), _use_mask)
            outs.append(_out)
            mvg = _out.warped_file
        out = _collate_outputs(outs) if collate else outs
        return out
    else:
        return _register(moving, parameters, output_path)


def _elastix_register_mp(args, **kwargs):
    """Reorder arguments for multiprocessing support."""
    moving, moving_mask, output_path = args
    return _elastix_register(
        moving=moving, moving_mask=moving_mask, output_path=output_path, **kwargs
    )


def _apply_warp(
    moving: MedVolOrPath,
    output_path: str = None,
    transform: Union[str, Sequence[str]] = None,
    out_registration: RegistrationOutputSpec = None,
    rtype: type = MedicalVolume,
    num_threads: int = 1,
    show_pbar: bool = False,
) -> MedVolOrPath:
    assert rtype in [MedicalVolume, str], rtype  # rtype must be MedicalVolume or str
    has_output_path = bool(output_path)
    if rtype == str and not has_output_path:
        raise ValueError("`output_path` must be specified when `rtype=str`")
    if not output_path:
        # TODO: Add path generation that prevents collisions during multiprocessing.
        # When multiprocessing executes rapidly and poor seed is set, the uuids have
        # collided. To avoid this, we append the process name to the directory that is
        # created.
        output_path = os.path.join(
            env.temp_dir(), f"apply_warp-{str(uuid.uuid1())}-{str(uuid.uuid4())}"
        )
        if not _is_main_process():
            output_path += mp.current_process().name
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    if not transform:
        transform = out_registration.transform
    elif isinstance(transform, str):
        transform = [transform]
    transform = [os.path.abspath(t) for t in transform]

    mv_filepath = os.path.join(output_path, "moving.nii.gz")
    if isinstance(moving, MedicalVolume):
        NiftiWriter().save(moving, mv_filepath)
        moving = mv_filepath

    transformix_path = _local_exe("transformix")  # noqa
    cwd = _local_lib_dir()
    for tf in tqdm(transform, disable=not show_pbar):
        reg = ApplyWarp()
        reg.inputs.moving_image = moving
        reg.inputs.transform_file = tf
        reg.inputs.output_path = output_path
        reg.terminal_output = preferences.nipype_logging
        reg.inputs.num_threads = num_threads
        reg_output = reg.run(cwd=cwd)

        moving = reg_output.outputs.warped_file

    if rtype == MedicalVolume:
        out = NiftiReader().load(moving)
    else:
        out = moving

    if os.path.isfile(mv_filepath):
        os.remove(mv_filepath)
    if not has_output_path and os.path.isdir(output_path):
        shutil.rmtree(output_path)

    return out


def _apply_warp_mp(args, **kwargs):
    """Reorder arguments for multiprocessing support."""
    moving, output_path = args
    return _apply_warp(moving=moving, output_path=output_path, **kwargs)


def _write(vol: MedicalVolume, path: str):
    """Extracted out for multiprocessing purposes."""
    NiftiWriter().save(vol, path)


def _read(path: str):
    return NiftiReader().load(path)


def _local_exe(exe):
    """Returns path to local executable if exists, else None."""
    assert exe in ["elastix", "transformix"]
    dosma_path = os.path.join(fc._DOSMA_ELASTIX_FOLDER, exe)
    if os.path.isfile(dosma_path):
        return os.path.abspath(dosma_path)


def _local_lib_dir():
    """Returns path to directory with local lib file if exists, else None."""
    files = [x for x in os.listdir(fc._DOSMA_ELASTIX_FOLDER) if x.startswith("libANNlib")]
    if len(files) > 0:
        return fc._DOSMA_ELASTIX_FOLDER


def _is_main_process():
    py_version = tuple(sys.version_info[0:2])
    return (py_version < (3, 8) and mp.current_process().name == "MainProcess") or (
        py_version >= (3, 8) and mp.parent_process() is None
    )
