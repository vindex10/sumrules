from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from cubature import cubature

from .parallel import npMap, pyMap
from .analytics import mom, energ, beta, eta, coAngle

from . import constants

def stackArgRes(args, res):
    isVec = True if len(args.shape) == 2 else False
    
    if not isVec and len(res.shape) == 0:
            return sp.hstack((args, sp.array((res,))))
    elif isVec and len(res.shape) == 1:
            return sp.hstack((args, res.reshape(res.shape[0], 1)))
    else:
            return sp.hstack((args, res))

class McolPEvaluator:
    def __init__(self, mp, psicolp):
        self.CONST = constants
        self.psiColP = psicolp
        self.MP = mp
        self.mapper = npMap
        self.vectorized = False
        self.monitor = None
        self.maxP = 500
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, paramdict=None):
        keylist = ("maxP"
                  ,"absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, p, q, Tpq, Fpq):
        res, err = cubature(self.McolP_f, 3, 2, [0, 0, 0], [self.maxP, sp.pi, 2*sp.pi], args=[p, q, Tpq], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, p, q, Tpq):
        """
            x_args:
                px[0] for px
                px[1] for theta(px,p)
                px[2] for phi
        """
        res = self.mapper(lambda px:\
                    sp.sin(px[1])\
                    *px[0]**2/(2*sp.pi)**3*(energ(p, self.CONST["m"])/energ(px[0], self.CONST["m"]))\
                    *sp.conj(self.psiColP(p, px[0], coAngle(sp.cos(Tpq), sp.cos(px[1]), px[2])))\
                    *self.MP(px[0], q, px[1], px[2])\
                , x_args)
        res = sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res))
        
        return res

class McolPDiscEvaluator:
    def __init__(self, mp, psicolp, denerg):
        self.CONST = constants
        self.psiColP = psicolp
        self.denerg = denerg
        self.MP = mp
        
        self.mapper = npMap
        self.vectorized = False
        self.monitor = None

        self.maxP = 1000
        # be careful with precision, I spent some time to find out its importans
        self.absErr = 1e-10
        self.relErr = 1e-8

    def params(self, paramdict=None):
        keylist = ("maxP"
                  ,"absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, n, l):
        res, err = cubature(self.McolP_f, 3, 2, [0, 0, 0], [self.maxP, sp.pi, 2*sp.pi], args=[n, l], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, n, l):
        """
            x_args:
                px[0] for px
                px[1] for theta(px,p)
                px[2] for phi
        """
        res = self.mapper(lambda px:\
                    sp.sin(px[1])\
                    *px[0]**2/(2*sp.pi)**3\
                    *sp.sqrt(self.denerg(n, l)/2)/energ(px[0], self.CONST["m"])\
                    *self.psiColP(n, l\
                                , px[0], px[1], px[2])\
                    *sp.conj(\
                        self.MP(px[0]\
                              , mom(self.denerg(n, l)**2)\
                              , px[1], px[2])\
                    )\
                , x_args)
        res = sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res))
        
        return res

class GammaEvaluator:
    def __init__(self, mp):
        self.CONST = constants
        self.MPEvaluatorInstance = mp
        self.denerg = self.MPEvaluatorInstance.denerg
        self.monitor = None
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, paramdict=None):
        keylist = ("absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, n, l):
        res = self.CONST["dimfactor"]*self.CONST["Nc"]/(2*l+1)\
                /(16*sp.pi*self.denerg(n, l))\
                *sp.absolute(self.MPEvaluatorInstance.compute(n, l))**2

        if self.monitor is not None:
            self.monitor.push(stackArgRes(sp.array((n , l)), res))

        return res

class SigmaEvaluator:
    def __init__(self, mp):
        self.CONST = constants
        self.MPEvaluatorInstance = mp
        self.mapper = npMap
        self.vectorized = False
        self.monitor = None
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, paramdict=None):
        keylist = ("absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, s):
        res, err = cubature(self.sigma_f, 2, 1, [0, 0], [sp.pi, 2*sp.pi], args=[s], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0]

    def sigma_f(self, x_args, s):
        """
            x_args:
                x_args[0] - Tpq
                x_args[1] - Fpq
        """
        # use dimfactor for absErr to be reasonable.
        res = self.mapper(lambda px:\
                sp.sin(px[0])\
                *self.CONST["dimfactor"]*self.CONST["Nc"]*beta(s)/64/sp.pi**2/s\
                *sp.absolute(\
                             self.MPEvaluatorInstance.compute(mom(s, self.CONST["m"])\
                                                            , mom(s)\
                                                            , px[0]\
                                                            , px[1])\
                            )**2\
                , x_args)

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res))

        return res

class SumruleEvaluator:
    def __init__(self, sigma):
        self.CONST = constants
        self.SigmaEvaluatorInstance = sigma
        self.mapper = npMap
        self.vectorized = False
        self.monitor = None
        self.absErr = 1e-5
        self.relErr = 1e-3
        self.minS = 4*self.CONST["m"]**2 + 0.01
        self.maxS = 1000

    def params(self, paramdict=None):
        keylist = ("absErr"
                  ,"relErr"
                  ,"minS"
                  ,"maxS")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self):
        res, err = cubature(self.sumrule_f, 1, 1, [self.minS], [self.maxS], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0]

    def sumrule_f(self, x_args):
        """
            x_args:
                px[0] - s
        """
        res = self.mapper(lambda s:\
                self.SigmaEvaluatorInstance.compute(s)/s, x_args)

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res))

        return res

class SumruleDiscEvaluator:
    def __init__(self, gamma):
        self.CONST = constants
        self.GammaEvaluatorInstance = gamma
        self.denerg = self.GammaEvaluatorInstance.denerg
        self.mapper = pyMap
        self.vectorized = False
        self.monitor = None
        self.nMax = 1

    def params(self, paramdict=None):
        keylist = ("nMax",)

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self):
        res = sum(self.mapper(\
                      lambda qn: 16*sp.pi**2*(2*qn[1]+1)\
                                /self.denerg(*qn)**3\
                                *self.GammaEvaluatorInstance.compute(*qn)\
                    , sp.array( [(n, l)\
                            for n in range(1, self.nMax+1)\
                            for l in range(n)
                      ])
                ))

        return res


class TrivialEvaluator:
    def __init__(self, func):
        self.func = func
        self.monitor = None

    def compute(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if self.monitor is not None:
            adaptedArgs = sp.column_stack(sp.broadcast(*args)).T
            self.monitor.push(stackArgRes(adaptedArgs, res))


        return res

    def params(self, paramdict=None):
        if paramdict is None:
            return dict()
        return True
