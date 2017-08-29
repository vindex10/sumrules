from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from cubature import cubature

from ..parallel import npMap, mpMap

from ..analytics import psiColP, sqedMP0 as MP0, sqedMP2 as MP2
from ..analytics import mom, energ, beta, eta, coAngle

from . import config as mconfig

class McolPEvaluator:
    def __init__(self):
        self.CONST = mconfig
        self.psiColP = psiColP
        self.MP = MP0
        self.mapper = npMap
        self.vectorized = True
        self.maxP = 500
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, paramdict=None):
        keylist = ("vectorized"
                  ,"maxP"
                  ,"absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, p, q, Cpq):
        """
            p (for params):
                p, q, Cpq
        """
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
    def __init__(self):
        self.CONST = mconfig
        self.psiColP = psiColP
        self.McolPEvaluatorInstance = McolPEvaluator()
        self.mapper = npMap
        self.vectorized = False
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, paramdict=None):
        keylist = ("vectorized"
                  ,"absErr"
                  ,"relErr")

        if paramdict is None:
            return { k: getattr(self, k) for k in keylist }

        for key, val in paramdict.items():
            if key in keylist:
                setattr(self, key, val)
        return True

    def compute(self, s):
        res, err = cubature(self.sigma_f, 1, 1, [-1], [1], args=[s], abserr=self.absErr, relerr=self.relErr, vectorized=self.vectorized)
        return res[0]

    def sigma_f(self, x_args, s):
        """
            x_args:
                x_args[0] - Cpq
        """
        px = x_args.T if self.vectorized else x_args

        # use dimfactor for absErr to be reasonable
        res = self.mapper(lambda x: self.CONST["dimfactor"]*beta(s)/32/sp.pi/s*sp.absolute(self.McolPEvaluatorInstance.compute(mom(s, self.CONST["m"]), mom(s), x))**2, px[0])

        return res.T if self.vectorized else res

class SumruleEvaluator:
    def __init__(self):
        self.CONST = mconfig
        self.SigmaEvaluatorInstance = SigmaEvaluator()
        self.mapper = mpMap
        self.vectorized = True
        self.absErr = 1e-5
        self.relErr = 1e-3
        self.minS = 4*self.CONST["m"]**2 + 0.01
        self.maxS = 1000

    def params(self, paramdict=None):
        keylist = ("vectorized"
                  ,"absErr"
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
        px = x_args.T if self.vectorized else x_args

        res = self.mapper(lambda s: self.SigmaEvaluatorInstance.compute(s)/s, px[0])

        return res.T if self.vectorized else res

