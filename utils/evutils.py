"""@file

Provide helpers to use in evaluators.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp

def stackArgRes(args, res, params=None):
    """ Combine cubature `x_args`, output vector and params into one array.

        `stackArgRes` comes handy when you want to push to Monitor exact coordinate of
        evaluated result. So, to define exact path, you have to mention
        not only `x_args` but also params with which `compute` was called.

        Example of usage:

            >>> class someEvaluator(Integrator):
            >>> ...
            >>>     def compute(param1):
            >>>         ...
            >>>         cubature(func, args=(param1,), ...)
            >>>         ...
            >>>     def func(x_args, param1):
            >>>         ...
            >>>         res = self.cubMap(otherFunc, x_args)
            >>>         self.monitor.push(stackArgRes(x_args, res, (param1,)))
          
        Args:
            args: provide `x_args` passed by cubature
            res: result of computation per each x_arg

        Kwargs:
            params: list of params on which `res` depends


        Returns:
            NumPy array of lines.

            For example:
                
                >>> x_args = np.array(
                >>>             ((1, 2, 3)
                >>>             ,(4, 5, 6)
                >>>             )
                >>>         )
                >>> params = (7, 8)
                >>> res = (14, 15)
                >>> 
                >>> stackArgRes(x_args, res, params)
                np.array((1, 2, 3, 7, 8, 14)
                         ,(4, 5, 6, 7, 8, 15))

    """
    isVec = len(args.shape) == 2
    
    if params is not None:
        newargs = sp.hstack((args, sp.tile(params, (args.shape[0], 1))))\
                if  isVec\
                else  sp.hstack((args, params))
    else:
        newargs = args

    if not isVec and len(res.shape) == 0:
            out = sp.hstack((newargs, sp.array((res,))))
    elif isVec and len(res.shape) == 1:
            out =  sp.hstack((newargs, res.reshape(res.shape[0], 1)))
    else:
            out = sp.hstack((newargs, res))

    return out

