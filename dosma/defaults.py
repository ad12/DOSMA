"""Default parameters.

This is the first file that is imported from this software - all initialization imports should occur here.

Please change depending on your preferences.
"""
import os

import matplotlib
import nested_lookup
import shutil
import yaml

from typing import Any

__all__ = ['preferences']

# Parse preferences file
_file_dirpath = os.path.dirname(os.path.abspath(__file__))
_internal_preferences_template_filepath = os.path.join(_file_dirpath, 'resources/templates/.preferences.yml')
_preferences_cmd_line_filepath = os.path.join(_file_dirpath, 'resources/templates/.preferences_cmd_line_schema.yml')

__preferences_filepath__ = os.path.join(_file_dirpath, 'resources/preferences.yml')

if not os.path.isfile(__preferences_filepath__):
    shutil.copyfile(_internal_preferences_template_filepath, __preferences_filepath__)

# Default rounding for I/O (dicom, nifti, etc) - DO NOT CHANGE
AFFINE_DECIMAL_PRECISION = 4
SCANNER_ORIGIN_DECIMAL_PRECISION = 4

DEFAULT_FONT_SIZE = 16
DEFAULT_TEXT_SPACING = DEFAULT_FONT_SIZE * 0.01


class _Preferences(object):
    """A pseudo-Singleton class implementation to track preferences.

    Do not instantiate this class. To modify/update preferences use the preferences module variable defined below.

    However, in the case this class is instantiated, a new object will be created, but all config properties will be
    shared among all instances. Meaning modifying the config in one instance will impact the preferences state in
    another instance.
    """
    _preferences_filepath = __preferences_filepath__
    __config = {}
    __key_list = []

    def __init__(self):
        # Load config and set information if config has not been initialized.
        if not self.__config:
            with open(self._preferences_filepath, 'r') as f:
                self.__config = yaml.load(f)
            matplotlib.rcParams.update(self.__config['visualization']['matplotlib']['rcParams'])

            # Store all preference keys.
            self.__key_list = self._unroll_keys(self.config, '')

    def _unroll_keys(self, subdict: dict, prefix: str) -> list:
        """Recursive method to unroll keys."""
        pref_keys = []
        keys = subdict.keys()
        for k in keys:
            prefix_or_key = "{}/{}".format(prefix, k)
            val = subdict[k]
            if type(val) is dict:
                pref_keys.extend(self._unroll_keys(val, prefix_or_key))
            else:
                pref_keys.append(prefix_or_key)

        return pref_keys

    @staticmethod
    def _get_prefixes(prefix: str = ''):
        if not prefix:
            return ()

        r_prefixes = []
        prefixes = prefix.split('/')
        for p in prefixes:
            if p == '':
                continue
            r_prefixes.append(p)

        return r_prefixes

    def get(self, key: str, prefix: str = ''):
        """Get preference.

        Args:
            key (str): Preference to peek. Can be full path preference.
            prefix (:obj:`str`, optional): Prefix defining which sub-config to search. For example, to access the key
                `visualization/rcParams`, this parameter should be `visualization`. Defaults to ''.

        Returns:
            The preference value.
        """
        return self.__get(self.__config, key, prefix)[0]

    def __get(self, b_dict: dict, key: str, prefix: str=''):
        """Get preference.

        Args:
            b_dict (dict): Dictionary to search.
            key (str): Preference to peek. Can be full path preference.
            prefix (:obj:`str`, optional): Prefix defining which sub-config to search. For example, to access the key
                `visualization/rcParams`, this parameter should be `visualization`. Defaults to ''.

        Returns:
            The preference value, sub-dictionary to search
        """
        p_keys = self._get_prefixes(key)
        key = p_keys[-1]
        k_prefixes = list(p_keys[:-1])

        prefixes = list(self._get_prefixes(prefix))
        prefixes.extend(k_prefixes)

        subdict = b_dict
        for p in prefixes:
            subdict = subdict[p]

        num_keys = nested_lookup.get_occurrence_of_key(subdict, key)

        if num_keys == 0:
            raise KeyError('Key not \'%s \' found in prefix \'%s\'' % (key, prefix))
        if num_keys > 1:
            raise KeyError('Multiple keys \'%s \'found in prefix \'%s\'. Provide a more specific prefix.' % (key,
                                                                                                             prefix))

        return nested_lookup.nested_lookup(key, subdict)[0], subdict

    def set(self, key: str, value: Any, prefix: str = ''):
        """Set preference.

        Args:
            key (str): Preference to peek. Can be full path preference.
            value (any): Value to set preference.
            prefix (:obj:`str`, optional): Prefix defining which sub-config to search. For example, to access the key
                `visualization/rcParams`, this parameter should be `visualization`. Defaults to ''.

        Returns:
            The preference value.
        """
        val, subdict = self.__get(self.__config, key, prefix)

        # type of new value has to be the same type as old value
        if type(value) != type(val):
            try:
                value = type(val)(value)
            except (ValueError, TypeError) as e:
                raise TypeError('could not convert %s to %s: %s' % (type(value), type(val), value))

        p_keys = self._get_prefixes(key)
        key = p_keys[-1]
        nested_lookup.nested_update(subdict, key, value, in_place=True)

        # if param is an rcParam, update matplotlib
        if key in self.config['visualization']['matplotlib']['rcParams'].keys():
            matplotlib.rcParams.update({key: value})

    def save(self):
        """Save preferences to file.

        Args:
            file_path (:obj:`str`, optional): File path to yml file. Defaults to local preferences file.
        """
        with open(self._preferences_filepath, 'w') as f:
            yaml.dump(self.__config, f, default_flow_style=False)

    @property
    def config(self):
        """Get preferences configuration."""
        return self.__config

    # Make certain preferences easily accessible through this class.
    @property
    def segmentation_batch_size(self) -> int:
        return self.get('/segmentation/batch.size')

    @property
    def visualization_use_vmax(self) -> bool:
        return self.get('/visualization/use.vmax')

    @property
    def mask_dilation_rate(self) -> float:
        return self.get('/registration/mask/dilation.rate')

    @property
    def mask_dilation_threshold(self) -> float:
        return self.get('/registration/mask/dilation.threshold')

    @property
    def fitting_r2_threshold(self):
        return self.get('/fitting/r2.threshold')

    @property
    def image_data_format(self):
        from dosma.data_io.format_io import ImageDataFormat
        return ImageDataFormat[self.get('/data/format')]

    def cmd_line_flags(self) -> dict:
        """Provide command line flags for changing preferences via command line.

        Not all preferences are listed here. Only those that should easily be set.

        All default values will be based on the current state of preferences, not the static state specified in yaml
            file.

        Returns:
            Preference keys with corresponding argparse kwarg dict
        """
        with open(_preferences_cmd_line_filepath) as f:
            cmd_line_config = yaml.load(f)

        cmd_line_dict = {}
        for k in self.__key_list:
            try:
                argparse_kwargs, _ = self.__get(cmd_line_config, k)
            except KeyError:
                continue

            argparse_kwargs['default'] = self.get(k)
            argparse_kwargs['type'] = eval(argparse_kwargs['type'])
            cmd_line_dict[k] = argparse_kwargs

        return cmd_line_dict

    def __str__(self):
        return str(self.__config)


preferences = _Preferences()

