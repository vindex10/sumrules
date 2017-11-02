"""@package ConfigManager
Centralized manager of objects' configs

The only thing object should have, to be managable, is a `params` method.
See its basic implementation in sumrules::Config
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import re

class ConfigManager(object):
    """ Implementation of a centralized config manager.

        Attributes:
            watching: dict which stores Label -> Object pairs.
            _cfgre: regular expression for cfg entry.
            _prefre: regular expression to detach prefix from cfg entry.
    """

    _prefre = re.compile("^([A-Z0-9]+)_")
    _cfgre = re.compile("([^=\s]+)\s*=\s*(?:\"(.+)\"|'(.+)'|([^\s]+))")

    def __init__(self):
        self.watching = dict()

    def register(self, module, prefix):
        """ Bind Label to Object.
            
            Args:
                module: object you want to register.
                prefix: label for object. Use it as prefix in cfg files.
            
            Returns:
                Nothing.
        """
        assert prefix not in self.watching.keys()
        assert self._prefre.match(prefix+"_")
        assert "params" in module.__dir__()
        self.watching.update({prefix: module})

    def forget(self, prefix):
        if prefix in self.watching.keys():
            del self.watching[prefix]
    
    def readEnv(self, prefix=None):
        """Read cfg entries from environment and update corresponding objects
            
            Kwargs:
                prefix: look for entries with specific prefix.

            Returns:
                Nothing.

            Example:
                
                ## example.py
                # We need a managable object
                >>> obj = Config({"foo": "bar"})
                >>> 
                >>> cmgr = ConfigManager()
                >>> cmgr.register(obj, "OBJ")
                >>> cmgr.readEnv()
                >>> 
                >>> obj.params()

                $ OBJ_foo="barbarian" python example.py
                {"foo", "barbarian"}
        """
        if prefix is None:
            for pref in self.watching.keys():
                self.readEnv(prefix=pref)
            return

        updict = dict()
        for key in self.watching[prefix].params().keys():
            entry = prefix+"_"+key
            if entry in os.environ.keys():
                updict.update({key: self._parseStr(os.environ[entry])})
        self.watching[prefix].params(updict)


    def readFile(self, filename, prefix=None):
        """Read cfg entries from file and update corresponding objects.
            
            Args:
                filename: name of file to read entries from.
                prefix: look for entries with specific prefix.

            Returns:
                Nothing.

            Examples:
                
                ## example.py
                # We need a managable object
                >>> obj = Config({"foo": "bar"})
                >>> 
                >>> cmgr = ConfigManager()
                >>> cmgr.register(obj, "OBJ")
                >>> cmgr.readFile("example.conf")
                >>> 
                >>> obj.params()

                ## example.conf
                OBJ_foo = barbarian

                $ python example.py
                {"foo", "barbarian"}
        """
        if prefix is None:
            for pref in self.watching.keys():
                self.readFile(filename, prefix=pref)
            return

        try:
            with open(filename) as f:
                updict = dict()
                for line in f:
                    matching = self._cfgre.match(line)
                    if not matching:
                        continue
                    pair = [v for v in self._cfgre.match(line).groups() if v is not None]
                    if len(pair) > 0:
                        parsed = self._entryToPair(pair[0])
                        if parsed and parsed[0] == prefix:
                            updict.update({parsed[1]: self._parseStr(pair[1])})
            self.watching[prefix].params(updict)
        except (FileNotFoundError, TypeError):
            pass

    def keys(self):
        for prefix in self.watching.keys():
            for key in self.watching[prefix].params():
                yield prefix+"_"+key

    def items(self):
        for prefix in self.watching.keys():
            for key, value in self.watching[prefix].params().items():
                yield (prefix+"_"+key, value)

    def __getitem__(self, entry):
        try:
            prefix, key = self._entryToPair(entry)
            return self.watching[prefix].params()[key]
        except TypeError:
            raise KeyError

    def __setitem__(self, entry, value):
        prefix, key = self._entryToPair(entry)
        self.watching[prefix].params({key: value})

    def __iter__(self):
        return self.items()

    def _entryToPair(self, entry):
        """ Convert config entry Key to Prefix -> ObjKey pair.
            
            Args:
                entry: a key of the config entry.

            Returns:
                tuple(Prefix, ObjKey). Tuple of prefix and corresponding
                field name.
           
            Examples:

                # Line in config
                OBJ_foo = bar

                # Entry(here)
                OBJ_foo

                # Returned value:
                tuple(OBJ, foo)
        """
        match = self._prefre.match(entry)
        if not match:
            return False
        prefix = match.groups()[0]
        return (prefix, entry[len(prefix)+1:])

    @staticmethod
    def _parseStr(a):
        """ Convert user input into a Python type.
            
            Try evaluate expression as a Python data structure, return `str`
            otherwise. `ast.literal_eval` is used under the hood, so one can
            even pass simple arithmetical expressions as an argument.

            Args:
                a: input value.

            Returns:
                * passed data structure if possible.
                * `str` otherwise.
        """
        try:
            return ast.literal_eval(a)
        except:
            return str(a)

