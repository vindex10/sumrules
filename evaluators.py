from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from cubature import cubature

from .parallel import npMap
from .analytics import mom, energ, beta, eta, coAngle

from . import constants

class McolPEvaluator:
    def __init__(self, mp, psicolp):
        self.CONST = constants
        self.psiColP = psicolp
        self.MP = mp
        self.mapper = npMap
        self.vectorized = False
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

    def compute(self, p, q, Cpq, Fpq):
        res, err = cubature(self.McolP_f, 3, 2, [0, -1, 0], [self.maxP, 1, 2*sp.pi], args=[p, q, Cpq], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, p, q, Cpq):
        """
            x_args:
                px[0] for px
                px[1] for Cos(px,p)
                px[2] for phi
        """
        res = self.mapper(lambda px: px[0]**2/(2*sp.pi)**3*(energ(p, self.CONST["m"])/energ(px[0], self.CONST["m"]))*sp.conj(self.psiColP(p, px[0], coAngle(Cpq, px[1], px[2])))*self.MP(px[0], q, px[1], px[2]), x_args)
        res = sp.array((sp.real(res), sp.imag(res))).T

        return res

class SigmaEvaluator:
    def __init__(self, mp):
        self.CONST = constants
        self.MPEvaluatorInstance = mp
        self.mapper = npMap
        self.vectorized = False
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
        res, err = cubature(self.sigma_f, 2, 1, [-1, 0], [1, 2*sp.pi], args=[s], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0]

    def sigma_f(self, x_args, s):
        """
            x_args:
                x_args[0] - Cpq
                x_args[1] - Fpq
        """
        # use dimfactor for absErr to be reasonable.
        # Assuming Fpq contributes only as phase => force Fpq = 0
        res = self.mapper(lambda px: self.CONST["dimfactor"]*self.CONST["Nc"]*beta(s)/64/sp.pi**2/s*sp.absolute(self.MPEvaluatorInstance.compute(mom(s, self.CONST["m"]), mom(s), px[0], px[1]))**2, x_args)
        return res

class SumruleEvaluator:
    def __init__(self, sigma):
        self.CONST = constants
        self.SigmaEvaluatorInstance = sigma
        self.mapper = npMap
        self.vectorized = False
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
        res = self.mapper(lambda s: self.SigmaEvaluatorInstance.compute(s)/s, x_args)

        return res

class TrivialEvaluator:
    def __init__(self, func):
        self.func = func

    def compute(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def params(self, paramdict=None):
        if paramdict is None:
            return dict()
        return True
