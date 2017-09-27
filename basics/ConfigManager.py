from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import re

class ConfigManager(object):
    _prefre = re.compile("^([A-Z0-9]+)_")
    _cfgre = re.compile("([^=\s]+)\s*=\s*(?:\"(.+)\"|'(.+)'|([^\s]+))")

    def __init__(self):
        self.watching = dict()

    def register(self, module, prefix):
        assert prefix not in self.watching.keys()
        assert self._prefre.match(prefix+"_")
        assert "params" in module.__dir__()
        self.watching.update({prefix: module})

    def forget(self, prefix):
        if prefix in self.watching.keys():
            del self.watching[prefix]
    
    def readEnv(self, prefix=None):
        if prefix is None:
            for pref in self.watching.keys():
                self.readEnv(prefix=pref)
            return

        updict = dict()
        for key in self.watching[prefix].params().keys():
            entry = prefix+"_"+key
            if entry in os.environ.keys():
                updict.update({key: self.parseStr(os.environ[entry])})
        self.watching[prefix].params(updict)


    def readFile(self, filename, prefix=None):
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
                            updict.update({parsed[1]: self.parseStr(pair[1])})
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
        prefix, key = self._entryToPair(entry)
        return self.watching[prefix].params()[key]

    def __setitem__(self, entry, value):
        prefix, key = self._entryToPair(entry)
        self.watching[prefix].params({key: value})

    def __iter__(self):
        return self.items()

    def _entryToPair(self, entry):
        match = self._prefre.match(entry)
        if not match:
            return False
        prefix = match.groups()[0]
        return (prefix, entry[len(prefix)+1:])

    @staticmethod
    def _parseStr(a):
        if a == "True":
            return True

        if a == "False":
            return False

        try:
            return int(a)
        except ValueError:
            pass

        try:
            return float(a)
        except ValueError:
            pass

        return a

