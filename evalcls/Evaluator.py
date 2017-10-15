"""@package Evaluator
Stores trivial implementation of evaluator

Evaluators needed to combine analytical functions and other evaluators
from sumrules::lib to produce numerical result. You can implement your
own evaluator by nesting it from trivial one or those from
sumrules::lib::evaluators.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from ..utils import parallel
from .. import constants

class Evaluator(object):
    """Trivial evaluator.
        
        It is not appropriate for immediate use. One should implement
        `compute()` method before.

        Attributes:
            CONST: stores global constants. By default sumrules::constants.
            monitor: monitoring object. Instance of
                sumrules::misc::Monitor::Monitor. Use it to log mediated
                compuation results.
            mapper: implementation of Map function, to be able to Map in
                a parallel way. Choose one of mappers
                from sumrules::utils::parallel.
            _keylist: stores list of attributes of class which `params()`
                method will be able to update. It is trivial here, but
                useful for more complicated evaluators.
    """

    def __init__(self):
        self.CONST = constants
        self.monitor = None
        self.mapper = parallel.pyMap
        self._keylist = list()

    def params(self, paramdict=None):
        """Update evaluator parameters or get current values
            
            Update parameters with `paramdict` or return current values
            when paramdict is undefined.

            Args:
                paramdict: provide a dict of {key: value} pairs to update with

            Returns:
                True.
        """
        attrs = set(self._keylist) & set(self.__dict__.keys())

        if paramdict is None:
            return { k: getattr(self, k) for k in attrs }

        for key, val in paramdict.items():
            if key in attrs:
                setattr(self, key, val)
        return True

