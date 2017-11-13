"""@file
Collection of mappers for parallel computations are listed here.

Usually they are needed as properties for
sumrules::evalcls::Integrator::Integrator.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp

import multiprocessing.dummy
from .. import config

class mpPool(object):
    numth = config["numThreads"]
    maxtask = config["maxTaskPerChild"]
    thepool = multiprocessing.dummy.Pool(config["numThreads"])

    @staticmethod
    def get():
        if mpPool.numth != config["numThreads"]\
           or mpPool.maxtask != config["maxTaskPerChild"]:
            mpPool.numth = config["numThreads"]
            mpPool.maxtask = config["maxTaskPerChild"]
            mpPool.thepool = multiprocessing\
                             .dummy\
                             .Pool(processes=mpPool.numth
                                  ,maxtaskperchild=mpPool.maxtask)
        return mpPool.thepool

def mpMap(f, data):
    """ Multiprocessing map.
        Use Python's multiprocessing module to map `f` onto `data`
        in prallel threads.

        Args:
            f: function to map.
            data: iterable to map onto.
        
        Returns:
            NumPy array of computed values.    
    """
    res = mpPool.get().map(f, data)
    return sp.hstack(res)

def npMap(f, data):
    """ NumPy map
        In fact, the function is applied through NumPy
        [broadcasting](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html).

        * Warning. Use it only when you know what you do.

        Args:
            f: function to map.
            data: iterable to map onto.
        
        Returns:
            NumPy array of computed values.    
    """
    return f(data.T).T

def pyMap(f, data):
    """ Python regular map.

        Regular Python map function, but with output converted into
        NumPy array.

        Args:
            f: function to map.
            data: iterable to map onto.
        
        Returns:
            NumPy array of computed values.    
    """
    return sp.hstack(list(map(f, data)))
