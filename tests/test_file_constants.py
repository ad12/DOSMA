import unittest

from dosma import file_constants as fc


class FileConstants(unittest.TestCase):
    def setUp(self):
        print("Testing: ", self._testMethodName)

    def test_reinit_variables(self):
        """Test dynamically adjusting default variables"""
        do, oo = fc.DEBUG, fc.NIPYPE_LOGGING

        fc.set_debug()

        dn, on = fc.DEBUG, fc.NIPYPE_LOGGING

        assert do == 0, do
        assert dn == 1, dn

        assert oo == "file_stderr", oo
        assert on == "stream", on
