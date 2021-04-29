import os
import unittest

import dosma
from dosma.defaults import preferences
from dosma.utils import env


class TestEnv(unittest.TestCase):
    def test_package_available(self):
        assert env.package_available("dosma")
        assert not env.package_available("blah")

    def test_get_version(self):
        version = tuple(int(x) for x in dosma.__version__.split("."))
        assert env.get_version("dosma") == version[:2]
        assert env.get_version("dosma", 3) == version

    def test_debug(self):
        os_env = os.environ.copy()

        env.debug(True)
        assert preferences.nipype_logging == "stream"

        env.debug(False)
        assert preferences.nipype_logging == "file_stderr"

        os.environ = os_env  # noqa: B003
