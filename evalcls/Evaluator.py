from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from ..utils import parallel
from .. import constants

class Evaluator(object):
    def __init__(self):
        self.CONST = constants
        self.monitor = None
        self.mapper = parallel.pyMap
        self._keylist = list()

    def params(self, paramdict=None):
        attrs = set(self._keylist) & set(self.__dict__.keys())

        if paramdict is None:
            return { k: getattr(self, k) for k in attrs }

        for key, val in paramdict.items():
            if key in attrs:
                setattr(self, key, val)
        return True

