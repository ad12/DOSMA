"""Test defaults and preferences
Files tested: defaults.py
"""

import collections
import os
import unittest
from shutil import copyfile

import nested_lookup

from dosma.defaults import _Preferences

# Duplicate the resources file
_test_preferences_sample_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'resources/preferences.yml')
_test_preferences_duplicate_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'resources/.test.preferences.yml')


class PreferencesMock(_Preferences):
    _preferences_filepath = _test_preferences_duplicate_filepath


class TestPreferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        copyfile(_test_preferences_sample_filepath, _test_preferences_duplicate_filepath)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(_test_preferences_duplicate_filepath):
            os.remove(_test_preferences_duplicate_filepath)

    def test_duplicate(self):
        """Test duplicate instances share same config."""
        a = PreferencesMock()
        b = PreferencesMock()

        assert a.config == b.config, "Configs must be the same dictionary."

        # Test with _Preferences to be certain
        a = _Preferences()
        b = _Preferences()

        assert a.config == b.config, "Configs must be the same dictionary."

    def test_get(self):
        a = PreferencesMock()

        # Raise error when key doesn't exist.
        with self.assertRaises(KeyError):
            a.get('sample-key')

        # Raise error when key is not specific.
        assert nested_lookup.get_occurrence_of_key(a.config, 'foo') > 1
        with self.assertRaises(KeyError):
            a.get('foo')

        # No error when lookup is specific.
        a.get('testing1/foo')

    def test_set(self):
        a = PreferencesMock()

        # Raise error when key doesn't exist.
        with self.assertRaises(KeyError):
            a.set('sample-key', 'bar')

        # Raise error when key is not specific.
        assert nested_lookup.get_occurrence_of_key(a.config, 'foo') > 1, "%s." % a.config
        with self.assertRaises(KeyError):
            a.set('foo', 100)

        # Raise error when value is not the same
        with self.assertRaises(TypeError):
            a.set('testing1/foo', 'bar')

        # Success when using full path or prefix kwarg
        a.set('testing1/foo', 50)
        assert a.get('testing1/foo') == 50, "Value mismatch: got %s, expected %s" % (a.get('testing1/foo'), 50)

        a.set('foo', 100, 'testing1')
        assert a.get('testing1/foo') == 100, "Value mismatch: got %s, expected %s" % (a.get('testing1/foo'), 100)

    def test_write(self):
        a = PreferencesMock()

        a.set('testing1/foo', 250)
        a.save()

        b = PreferencesMock()

        assert a.config == b.config, "Configs must be the same dictionary."
        assert a.get('testing1/foo') == b.get('testing1/foo')


class TestPreferencesSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        copyfile(_test_preferences_sample_filepath, _test_preferences_duplicate_filepath)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(_test_preferences_duplicate_filepath):
            os.remove(_test_preferences_duplicate_filepath)

    def test_cmd_line_schema(self):
        """Test that the command line schema for preferences is valid.

        Checks:
            - No overlapping aliases
            - Fields ['aliases', 'type', 'nargs', 'help'] present
            - All aliases are list and begin with '--'
        """
        a = PreferencesMock()
        config_dict = a.cmd_line_flags()

        # Check to see not duplicates in aliases
        aliases = []
        for k in config_dict.keys():
            aliases.extend(config_dict[k]['aliases'])

        alias_duplicates = [item for item, count in collections.Counter(aliases).items() if count > 1]
        assert len(alias_duplicates) == 0, 'Duplicate aliases: %s' % alias_duplicates

        for k in config_dict.keys():
            arg = config_dict[k]
            # Check to see each field has at least 4 primary keys
            for field in ['name', 'aliases', 'type', 'nargs', 'help']:
                assert field in arg.keys(), '`%s` missing from %s' % (field, k)

            # Check type(aliases) is list
            assert type(arg['aliases']) is list, 'Aliases must be list - k' % k

            # Check to see each field has at least 4 primary keys
            for alias in arg['aliases']:
                assert alias.startswith('--'), 'Alias \'%s\' in %s must start with \'--\'' % (alias, k)

