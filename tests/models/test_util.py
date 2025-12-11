import unittest

from dosma.models import model_loading_util as m_util


class TestUtil(unittest.TestCase):
    def test_aliases_exist(self):
        # Verify that none of the supported segmentation models have overlapping aliases
        models = m_util.__SUPPORTED_MODELS__
        alias_to_model = {}  # noqa: F841

        for m in models:
            aliases = m.ALIASES

            # all supported models must have at least 1 alias that is not ''
            valid_alias = len(aliases) >= 1 and all(x != "" for x in aliases)

            assert valid_alias, "%s does not have valid aliases" % m

    def test_overlapping_aliases(self):
        # Verify that none of the supported segmentation models have overlapping aliases
        models = m_util.__SUPPORTED_MODELS__
        alias_to_model = {}

        for m in models:
            curr_aliases = set(alias_to_model.keys())
            aliases = set(m.ALIASES)

            assert aliases.isdisjoint(curr_aliases), "%s alias(es) already in use" % str(
                aliases.intersection(curr_aliases)
            )

            for a in aliases:
                alias_to_model[a] = m
