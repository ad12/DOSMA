.. _utils_api:

Utilities
================================================================================

Collect Env
---------------------------
.. _utils_api_collect_env:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.utils.collect_env.collect_env_info


Env
---------------------------
.. _utils_api_env:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.debug
   dosma.utils.env.package_available
   dosma.utils.env.get_version


Logger
---------------------------
.. _utils_api_logger:

.. autosummary::
   :toctree: generated
   :nosignatures:

   dosma.setup_logger

If you do not want logging messages to display on your console (terminal, Jupyter Notebook, etc.),
the code below will only log messages at the ERROR level or higher:

>>> import logging
>>> dm.setup_logger(stream_lvl=logging.ERROR)
