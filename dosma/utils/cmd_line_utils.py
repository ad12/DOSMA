__all__ = ["ActionWrapper"]


class ActionWrapper(object):
    """Wrapper for actions (methods) that can be executed via command-line.

    Examples include `segment` scans, `interregister` scans, etc.

    Actions are instance methods of classes that can be executed via the command line.
    They are typically associated with different scans.

    To expose these methods to the command-line interface, we wrap these actions as subparsers.
        Parameters for the method are arguments of the subparser.
    """

    def __init__(self, name, **kwargs):
        self._name = name
        self._help = ""
        self._param_help = None
        self._alternative_param_names = None
        self._aliases = []

        if "help" in kwargs:
            self._help = kwargs.get("help")

        if "aliases" in kwargs:
            aliases = kwargs.get("aliases")
            assert type(aliases) is list, "aliases must be a list"
            self._aliases = aliases

        if "param_help" in kwargs:
            param_help_in = kwargs.get("param_help")
            assert type(param_help_in) is dict, "`param_help` must be a dictionary of str->str"
            for param_name in param_help_in:
                assert type(param_name) is str, "Keys must be of string type"
                assert type(param_help_in[param_name]) is str, "Values must be of string type"
            self._param_help = dict(param_help_in)

        if "alternative_param_names" in kwargs:
            alternative_param_names_in = kwargs.get("alternative_param_names")
            assert (
                type(alternative_param_names_in) is dict
            ), "`alternative_param_names` must be a dictionary of str->str"
            for param_name in alternative_param_names_in:
                assert type(param_name) is str, "Keys must be of string type"
                assert type(alternative_param_names_in[param_name]) in [
                    list,
                    tuple,
                ], "Values must be of string type"
            self._alternative_param_names = alternative_param_names_in

    def get_alternative_param_names(self, param: str):
        """Get aliases (alternate names) for a parameter.

        Args:
            param (str): Action parameter name.

        Returns:
            Optional[list[str]]: If aliases exist for parameter. `None`, otherwise.
        """
        if not self._alternative_param_names or param not in self._alternative_param_names:
            return None

        return self._alternative_param_names[param]

    def get_param_help(self, param: str):
        """Get help menu for a parameter.

        Args:
            param (str): Action parameter name.

        Returns:
            str: Help menu for parameter, if initialized. `""`, otherwise.
        """
        if not self._param_help or param not in self._param_help:
            return ""

        return self._param_help[param]

    @property
    def aliases(self):
        """list[str]: Aliases (other names) for this action."""
        return self._aliases

    @property
    def help(self):
        """str: Help menu for this action."""
        return self._help

    @property
    def name(self):
        """str: Action name."""
        return self._name
