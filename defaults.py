"""Default parameters
Please change depending on your preferences

This is the first file that is imported from this software - all initialization imports should occur here
"""
import os

import nested_lookup
import yaml

from data_io.format_io import ImageDataFormat

# Parse preferences file
__preferences_filepath__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources/preferences.yml')


class _Preferences():
    """A pseudo-Singleton class implementation to track preferences.

    Do not instantiate this class. To modify/update preferences use the preferences module variable defined below.

    However, in the case this class is instantiated, a new object will be created, but all config properties will be
    shared among all instances. Meaning modifying the config in one instance will impact the preferences state in
    another instance.
    """
    _preferences_filepath = __preferences_filepath__
    __config = {}

    def __init__(self):
        # Load config and set information if config has not been initialized
        if not self.__config:
            with open(self._preferences_filepath, 'r') as f:
                self.__config = yaml.load(f)
            matplotlib.use(self.__config['visualization']['matplotlib']['rcParams']['backend'])
            matplotlib.rcParams.update(self.__config['visualization']['matplotlib']['rcParams'])

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

    def get(self, key, prefix: str = ''):
        """Get preference.
        :param key: Preference to peek. Can be full path preference.
        :type key: str
        :param prefix: prefix defining which sub-config to search (e.g. 'visualization/rcParams), defaults to ''
        :type prefix: str, optional

        :return: the preference.
        """
        return self.__get(key, prefix)[0]

    def __get(self, key, prefix=''):
        """Get preference.
        :param key: Preference to peek. Can be full path preference.
        :type key: str
        :param prefix: prefix defining which sub-config to search (e.g. 'visualization/rcParams), defaults to ''
        :type prefix: str, optional

        :return: the preference value, the subdictionary to search
        :rtype tuple of lenght 2
        """
        p_keys = self._get_prefixes(key)
        key = p_keys[-1]
        k_prefixes = list(p_keys[:-1])

        prefixes = list(self._get_prefixes(prefix))
        prefixes.extend(k_prefixes)

        subdict = self.__config
        for p in prefixes:
            subdict = subdict[p]

        num_keys = nested_lookup.get_occurrence_of_key(subdict, key)

        if num_keys == 0:
            raise KeyError('Key not found in prefix \'%s\'' % key)
        if num_keys > 1:
            raise KeyError('Multiple keys \'%s \'found in prefix \'%s\'. Provide a more specific prefix.' % (key,
                                                                                                             prefix))

        return nested_lookup.nested_lookup(key, subdict)[0], subdict

    def set(self, key, value, prefix: str = ''):
        """Set preference.
        :param key: Preference to peek
        :type key: str
        :param value: value to set prefix.
        :type value: type of existing value
        :param prefix: prefix defining which sub-config to search (e.g. 'visualization/rcParams), defaults to ''
        :type prefix: str, optional

        :return: the preference.
        """
        val, subdict = self.__get(key, prefix)

        # type of new value has to be the same type as old value
        if type(value) != type(val):
            raise TypeError('Value of type %s, expected type %s' % (type(value), type(val)))

        p_keys = self._get_prefixes(key)
        key = p_keys[-1]
        nested_lookup.nested_update(subdict, key, value, in_place=True)

        # if param is an rcParam, update matplotlib
        if key in self.config['visualization']['matplotlib']['rcParams'].keys():
            matplotlib.rcParams.update({key: value})

    def save(self):
        with open(self._preferences_filepath, 'w') as f:
            yaml.dump(self.__config, f, default_flow_style=False)

    @property
    def config(self):
        """Get preferences configuration."""
        return self.__config

    # Make certain properties easily accessible through this class
    @property
    def segmentation_batch_size(self):
        return self.get('/segmentation/batch.size')

    @property
    def visualization_use_vmax(self):
        return self.get('/visualization/use.vmax')

    @property
    def mask_dilation_rate(self):
        return self.get('/registration/mask/dilation.rate')

    @property
    def mask_dilation_threshold(self):
        return self.get('/registration/mask/dilation.threshold')

    @property
    def fitting_r2_threshold(self):
        return self.get('/fitting/r2.threshold')

    @property
    def image_data_format(self):
        return ImageDataFormat[self.get('/data/format')]


preferences = _Preferences()

# Default rounding for I/O (dicom, nifti, etc) - DO NOT CHANGE
AFFINE_DECIMAL_PRECISION = 4
SCANNER_ORIGIN_DECIMAL_PRECISION = 4

DEFAULT_FONT_SIZE = 16
DEFAULT_TEXT_SPACING = DEFAULT_FONT_SIZE * 0.01

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': DEFAULT_FONT_SIZE})

DEFAULT_FIG_FORMAT = 'png'
