"""@package Config
Stores class Config
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

class Config(dict):
    """ Dict appropriate for usage with sumrules::misc::ConfigManager::ConfigManager.
        
        It is a basic implementation of an object managable by
        sumrules::misc::ConfigManager. See more examples in
        sumrules::lib::evaluators.
    """

    def params(self, paramdict=None):
        """ Manages dict values for sumrules::misc::ConfigManager::ConfigManager.

            Method allows to update existing values through `paramdict`
            argument, or get current values by calling it without arguments.
            
            Kwargs:
                paramdict: Dict of params with new values to update with

            Examples:

                >>> cfg = Config({"a": 10})
                >>> cfg.params()
                {"a": 10}

                >>> cfg.params({"a": 8})
                >>> cfg.params()
                {"a": 8}
                
        """
        if paramdict is None:
            return dict(self.items())
        else:
            for key, value in paramdict.items():
                if key in self.keys():
                    self.update({key: value})

