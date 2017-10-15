"""@package Integrator
Stores Evaluator adapted to use with `cubature`

[cubature](http://saullocastro.github.io/cubature/#cubature.cubature)
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp

from .Evaluator import Evaluator
from .. import constants

class Integrator(Evaluator):
    """ Evaluator with helpers for convenient use of `cubature` package.
        Attributes:
            vectorized: Corresponds to the `vectorized` flag in
                [cubature](http://saullocastro.github.io/cubature/#cubature.cubature)
            area: NumPy array of ranges to integrate in.

                Example (integrate over a ball of radius 10):

                    >>> self.area = np.array(
                    >>>     ((0, 10) # radial coordinate
                    >>>     ,(0, np.pi) # azimuthal angle
                    >>>     ,(0, 2*np.pi) # polar angle
                    >>>     )
                    >>> )

            cyclics: Define coordinates from `area`, on which
                function doesn't depends. It is a dict, where key corresponds
                to ordinal position of coordinate and value - point from where
                to take value.

                Example (Axial-symmetric function):

                    >>> self.cyclics = {2: np.pi}
                
                Integration over this coordinate will be ommited, and
                answer will be multiplied by length of the interval
                (here 2*np.pi)
    """

    def __init__(self):
        super(Integrator, self).__init__()
        self.vectorized = False
        self.area = list()
        self.cyclics = dict()

    def areaCyclics(self):
        """ Prepare integration area taking into account cyclics
            
            Returns:
                NumPy array of intervals, same as self.area, ommiting cyclics
        """
        return sp.delete(self.area, list(self.cyclics.keys()), axis=0)
    
    def cubMap(self, func, args):
        """ Special mapper which takes into account vectorization.
            
            Cubature provides different `x_args` depending on `vectorized`
            flag. So mapping should be applied in corresponding ways.

            Args:
                func: function to map
                args: iterable to map `func` onto
                    
                    * it is usual behavior, but everything depends on
                    defined mapper in self.mapper. For example:
                    sumrules::utils::parallel::npMap() has another behavior.

            Returns:
                List of values, not a map object.
        """
        if self.vectorized:
            return self.mapper(func, args)
        else:
            return func(args)

    def xargsCyclics(self, x_args):
        """ Prepare x_args taking into account cyclics.
            
            `xargsCyclics` is used inside of function which is passed to 
            `cubature` integrator. While we ommiting some coordinates due
            accounting cyclics, the function should be able to compute
            values for any valid `x_args`. The easiest workaround is to
            restore cyclic coordinates before using `x_args` for computations.

            Returns:
                NumPy array  of cubature `x_args` with restored cyclic coordinates.
        """
        isVec = len(x_args.shape) == 2
        extShape = x_args.shape if isVec else x_args.shape+(1,)
        args = x_args.T if isVec else x_args

        for i in range(len(self.area)):
            if i in self.cyclics.keys():
                args = sp.insert(args, i, sp.tile(self.cyclics[i], extShape[1]), axis=0)
        
        return args.T if isVec else args
    
    def cyclicPrefactor(self):
        """ Compute prefactor for cyclic coordinates.
            
            Multiply widths of all ranges which have been marked as cyclic.

            Returns:
                Float number, representing missed volume.
        """
        out = 1
        for i in self.cyclics.keys():
            out *= self.area[i, 1] - self.area[i, 0]
        
        return out
