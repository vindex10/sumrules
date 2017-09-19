from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp

def stackArgRes(args, res, params=None):
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

