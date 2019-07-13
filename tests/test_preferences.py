"""Test defaults and preferences
Files tested: defaults.py
"""

import os
import unittest
from shutil import copyfile

import nested_lookup

from defaults import _Preferences

# Duplicate the resources file
_test_preferences_sample_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'resources/preferences.yml')
_test_preferences_duplicate_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'resources/.preferences.yml')
copyfile(_test_preferences_sample_filepath, _test_preferences_duplicate_filepath)


class PreferencesMock(_Preferences):
    _preferences_filepath = _test_preferences_duplicate_filepath


class TestPreferences(unittest.TestCase):
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
