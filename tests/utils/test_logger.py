from dosma.utils.logger import setup_logger
from dosma.utils import env
import unittest
import logging

from .. import util


class TestSetupLogger(unittest.TestCase):
    def test_log_info(self):
        debug_val = env.debug()

        env.debug(False)
        setup_logger(None)
        with self.assertLogs("dosma", level="INFO"):
            logging.getLogger("dosma").info("Sample log at INFO level")

        env.debug(True)
        setup_logger(util.TEMP_PATH)
        with self.assertLogs("dosma", level="DEBUG"):
            logging.getLogger("dosma").debug("Sample log at DEBUG level")

        env.debug(debug_val)
