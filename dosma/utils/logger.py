"""Logging utility.
"""

import atexit
import functools
import logging
import os
import sys
from time import localtime, strftime
from typing import Union

from dosma.utils import env

from termcolor import colored

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
    *,
    color=True,
    name="dosma",
    abbrev_name=None,
    level=None,
):
    """Initialize the dosma logger.

    Args:
        output (str | bool): A file name or a directory to save log. If True, will save to
            the default dosma log location.
            If ``None`` or ``False``, will not save log file. If ends with ".txt" or ".log",
            assumed to be a file name. Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    if level is None:
        level = logging.DEBUG if env.debug() else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear handlers if they exist.
    is_new_logger = not logger.hasHandlers()
    if not is_new_logger:
        logger.handlers.clear()

    if abbrev_name is None:
        abbrev_name = "dm" if name == "dosma" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
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
            filename = os.path.join(output, "log.txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    if is_new_logger and name == "dosma":
        logger.info("\n" * 4)
        logger.info("==" * 40)
        logger.info("Timestamp: %s" % strftime("%Y-%m-%d %H:%M:%S", localtime()))
        logger.info("\n\n")

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
