import unittest

from dosma.utils.collect_env import collect_env_info


class TestCollectEnvInfo(unittest.TestCase):
    def test_collect_env_info(self):
        env_info = collect_env_info()
        assert isinstance(env_info, str)
