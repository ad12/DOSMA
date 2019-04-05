class ActionWrapper():
    """Wrapper for actions (methods) that can be executed via command-line"""

    def __init__(self, name, **kwargs):
        self.name = name
        self._help = ''
        self._param_help = None
        self._alternative_param_names = None
        self._aliases = []

        if 'help' in kwargs:
            help = kwargs.get('help')
            self._help = help

        if 'aliases' in kwargs:
            aliases = kwargs.get('aliases')
            assert type(aliases) is list, "aliases must be a list"
            self._aliases = aliases

        if 'param_help' in kwargs:
            param_help_in = kwargs.get('param_help')
            assert type(param_help_in) is dict, "param_help must be a dictionary of str->str"
            for param_name in param_help_in:
                assert type(param_name) is str, "keys must be of string type"
                assert type(param_help_in[param_name]) is str, "values must be of string type"
            self._param_help = dict(param_help_in)

        if 'alternative_param_names' in kwargs:
            alternative_param_names_in = kwargs.get('alternative_param_names')
            assert type(alternative_param_names_in) is dict, "param_help must be a dictionary of str->str"
            for param_name in alternative_param_names_in:
                assert type(param_name) is str, "keys must be of string type"
                assert type(alternative_param_names_in[param_name]) in [list, tuple], "values must be of string type"
            self._alternative_param_names = alternative_param_names_in

    def get_alternative_param_names(self, param: str):
        if not self._alternative_param_names or param not in self._alternative_param_names:
            return None

        return self._alternative_param_names[param]

    def get_param_help(self, param: str):
        if param not in self._param_help:
            return ''

        return self._param_help[param]

    @property
    def aliases(self):
        return self._aliases

    @property
    def help(self):
        return self._help


if __name__ == '__main__':
    a = ActionWrapper(name='segment', param_help={'hezbula': 'hexb'}, alternative_param_names={'hezbula': ['hexb']})
    print(a)
