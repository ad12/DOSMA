"""Logging utility.
"""

import atexit
import functools
import logging
import os
import sys
from time import localtime, strftime
from typing import Union

from termcolor import colored

from dosma.utils import env

__all__ = ["setup_logger"]


class _ColorfulFormatter(logging.Formatter):
    """
    This class is adapted from Facebook's detectron2:
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py
    """

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    output: Union[str, bool] = True,
    color=True,
    name="dosma",
    abbrev_name=None,
    stream_lvl=None,
    overwrite_handlers: bool = False,
):
    """Initialize the dosma logger.

    Args:
        output (str | bool): A file name or a directory to save log or a boolean.
            If ``True``, logs will save to the default dosma log location
            (:func:`dosma.utils.env.log_file_path`).
            If ``None`` or ``False``, logs will not be written to a file. This is not recommended.
            If a string and ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        color (bool): If ``True``, logs printed to terminal (stdout) will be in color.
        name (str): The root module name of this logger.
        abbrev_name (str): An abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "dosma" to "dm" and leave other
            modules unchanged.
        stream_lvl (int): The level for logging to console. Defaults to ``logging.DEBUG``
            if :func:`dosma.utils.env.debug()` is ``True``, else defaults to ``logging.INFO``.
        overwrite_handlers (bool): It ``True`` and logger with name ``name`` has logging handlers,
            these handlers will be removed before adding the new handlers. This is useful
            when to avoid having too many handlers for a logger.

    Returns:
        logging.Logger: A logger.

    Note:
        This method removes existing handlers from the logger.

    Examples:
        >>> setup_logger()  # how initializing logger is done most of the time
        >>> setup_logger("/path/to/save/dosma.log")  # save log to particular file
        >>> setup_logger(
        ... stream_lvl=logging.WARNING,
        ... overwrite_handlers=True)  # only prints warnings to console
    """
    if stream_lvl is None:
        stream_lvl = logging.DEBUG if env.debug() else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear handlers if they exist.
    is_new_logger = not logger.hasHandlers()
    if not is_new_logger and overwrite_handlers:
        logger.handlers.clear()

    if abbrev_name is None:
        abbrev_name = "dm" if name == "dosma" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(stream_lvl)
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output is not None and output is not False:
        if output is True:
            output = env.log_file_path()

        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "dosma.log")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    if is_new_logger and name == "dosma":
        logger.debug("\n" * 4)
        logger.debug("==" * 40)
        logger.debug("Timestamp: %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
        logger.debug("\n\n")

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
