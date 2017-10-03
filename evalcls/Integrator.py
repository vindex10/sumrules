from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp

from .Evaluator import Evaluator
from .. import constants

class Integrator(Evaluator):
    def __init__(self):
        super(Integrator, self).__init__()
        self.vectorized = False
        self.area = list()
        self.cyclics = dict()

    def areaCyclics(self):
        return sp.delete(self.area, list(self.cyclics.keys()), axis=0)
    
    def cubMap(self, func, args):
        if self.vectorized:
            return self.mapper(func, args)
        else:
            return func(args)

    def xargsCyclics(self, x_args):
        isVec = len(x_args.shape) == 2
        extShape = x_args.shape if isVec else x_args.shape+(1,)
        args = x_args.T if isVec else x_args

        for i in range(len(self.area)):
            if i in self.cyclics.keys():
                args = sp.insert(args, i, sp.tile(self.cyclics[i], extShape[1]), axis=0)
        
        return args.T if isVec else args
    
    def cyclicPrefactor(self):
        out = 1
        for i in self.cyclics.keys():
            out *= self.area[i, 1] - self.area[i, 0]
        
        return out
