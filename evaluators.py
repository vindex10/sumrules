from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from cubature import cubature

from .basics import BasicEvaluator, BasicIntegrator
from .utils.evutils import stackArgRes

from .utils.parallel import npMap, pyMap
from .analytics import mom, energ, beta, eta, coAngle

from . import constants

class McolPEvaluator(BasicIntegrator):
    def __init__(self, mp, psicolp):
        super(McolPEvaluator, self).__init__()
        self._keylist += ["maxP"
                         ,"absErr"
                         ,"relErr"]

        self.vectorized = False

        self.area = sp.array([[0, 500], [0, sp.pi], [0, 2*sp.pi]])
        self.psiColP = psicolp
        self.MP = mp

        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, params=None):
        suparams = super(McolPEvaluator, self).params(params)

        if params is None:
            suparams.update({"maxP": self.area[0][1]})
            return suparams

        if "maxP" in params.keys():
            self.area[0][1] = params["maxP"]
        
        return True

    def compute(self, p, q, Tpq, Fpq):
        area = self.areaCyclics()

        res, err = cubature(self.McolP_f\
                          , area.shape[0], 2\
                          , area.T[0]\
                          , area.T[1]\
                          , args=[p, q, Tpq]\
                          , abserr=self.absErr, relerr=self.relErr\
                          , vectorized=self.vectorized)

        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, p, q, Tpq):
        """
            x_args:
                px[0] for px
                px[1] for theta(px,p)
                px[2] for phi
        """
        x_args = self.xargsCyclics(x_args)

        res = self.mapper(lambda px:\
                    sp.sin(px[1])\
                    *px[0]**2/(2*sp.pi)**3*(energ(p, self.CONST["m"])/energ(px[0], self.CONST["m"]))\
                    *sp.conj(self.psiColP(p, px[0], coAngle(sp.cos(Tpq), sp.cos(px[1]), px[2])))\
                    *self.MP(px[0], q, px[1], px[2])\
                , x_args)
        res = self.cyclicPrefactor()*sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res, sp.array((p, q, Tpq))))
        
        return res


class McolPDiscEvaluator(BasicIntegrator):
    def __init__(self, mp, psicolp, denerg):
        super(McolPDiscEvaluator, self).__init__()
        self._keylist += ["maxP"
                         ,"absErr"
                         ,"relErr"]

        self.vectorized = False

        self.psiColP = psicolp
        self.denerg = denerg
        self.MP = mp
        
        self.area = sp.array([[0, 500], [0, sp.pi], [0, 2*sp.pi]])
        # be careful with precision, I spent some time to find out its importance
        self.absErr = 1e-10
        self.relErr = 1e-8

    def params(self, params=None):
        suparams = super(McolPDiscEvaluator, self).params(params)

        if params is None:
            suparams.update({"maxP": self.area[0][1]})
            return suparams

        if "maxP" in params.keys():
            self.area[0][1] = params["maxP"]
        
        return True

    def compute(self, n, l):
        area = self.areaCyclics()

        res, err = cubature(self.McolP_f\
                          , area.shape[0], 2\
                          , area.T[0]\
                          , area.T[1]\
                          , args=[n, l]\
                          , abserr=self.absErr, relerr=self.relErr\
                          , vectorized=self.vectorized)
        
        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, n, l):
        """
            x_args:
                px[0] for px
                px[1] for theta(px,p)
                px[2] for phi
        """
        x_args = self.xargsCyclics(x_args)

        res = self.cyclicPrefactor()*self.mapper(lambda px:\
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
        res = self.cyclicPrefactor()*sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res, sp.array((n,l))))
        
        return res


class GammaDiscEvaluator(BasicEvaluator):
    def __init__(self, mp):
        super(GammaDiscEvaluator, self).__init__()
        self.MPEvaluatorInstance = mp
        self.denerg = self.MPEvaluatorInstance.denerg

    def compute(self, n, l):
        res = self.CONST["dimfactor"]*self.CONST["Nc"]/(2*l+1)\
                /(16*sp.pi*self.denerg(n, l))\
                *sp.absolute(self.MPEvaluatorInstance.compute(n, l))**2

        if self.monitor is not None:
            self.monitor.push(stackArgRes(sp.array((n , l)), res))

        return res


class SigmaEvaluator(BasicIntegrator):
    def __init__(self, mp):
        super(SigmaEvaluator, self).__init__()
        self._keylist += ["absErr"
                         ,"relErr"]

        self.vectorized = False
        
        self.area = sp.array(((0, sp.pi), (0, 2*sp.pi)))
        self.MPEvaluatorInstance = mp

        self.absErr = 1e-5
        self.relErr = 1e-3

    def compute(self, s):
        area = self.areaCyclics()

        res, err = cubature(self.sigma_f\
                          , area.shape[0], 1\
                          , area.T[0]\
                          , area.T[1]\
                          , args=[s]\
                          , abserr=self.absErr, relerr=self.relErr\
                          , vectorized=self.vectorized)
        
        return res[0]

    def sigma_f(self, x_args, s):
        """
            x_args:
                x_args[0] - Tpq
                x_args[1] - Fpq
        """
        x_args = self.xargsCyclics(x_args)

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
        res *= self.cyclicPrefactor()

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res, sp.array((s,))))

        return res


class SumruleEvaluator(BasicIntegrator):
    def __init__(self, sigma):
        super(SumruleEvaluator, self).__init__()
        self._keylist += ["absErr"
                         ,"relErr"
                         ,"minS"
                         ,"maxS"]

        self.vectorized = False
        
        self.area = sp.array([[4*self.CONST["m"]**2 + 0.01
                             ,1000]])
        self.SigmaEvaluatorInstance = sigma
        
        self.absErr = 1e-5
        self.relErr = 1e-3

    def params(self, params=None):
        suparams = super(SumruleEvaluator, self).params(params)

        if params is None:
            suparams.update({"minS": self.area[0][0]
                            ,"maxS": self.area[0][1]})
            return suparams

        if "minS" in params.keys():
            self.area[0][0] = params["minS"]
        if "maxS" in params.keys():
            self.area[0][1] = params["maxS"]
        
        return True

    def compute(self):
        area = self.areaCyclics()
        res, err = cubature(self.sumrule_f\
                          , area.shape[0], 1\
                          , area.T[0]\
                          , area.T[1]\
                          , abserr=self.absErr, relerr=self.relErr\
                          , vectorized=self.vectorized)
        return res[0]

    def sumrule_f(self, x_args):
        """
            x_args:
                px[0] - s
        """
        x_args = self.xargsCyclics(x_args)

        res = self.mapper(lambda px:\
                self.SigmaEvaluatorInstance.compute(px[0])/px[0], x_args)
        res *= self.cyclicPrefactor()

        if self.monitor is not None:
            self.monitor.push(stackArgRes(x_args, res))

        return res


class SumruleDiscEvaluator(BasicEvaluator):
    def __init__(self, gamma):
        super(SumruleDiscEvaluator, self).__init__()
        self._keylist += ["nMax"]

        self.vectorized = False

        self.GammaEvaluatorInstance = gamma
        self.denerg = self.GammaEvaluatorInstance.denerg

        self.nMax = 1

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


class TrivialEvaluator(BasicEvaluator):
    def __init__(self, func):
        super(TrivialEvaluator, self).__init__()
        self.func = func

    def compute(self, *args, **kwargs):
        res = self.func(*args, **kwargs)

        if self.monitor is not None:
            adaptedArgs = sp.column_stack(sp.broadcast(*args)).T
            self.monitor.push(stackArgRes(adaptedArgs, res))

        return res

