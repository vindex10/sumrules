"""@file
Stores implementations of a variety of evaluators.

Nested from sumrules::evalcls.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from cubature import cubature

from ..evalcls.Evaluator import Evaluator
from ..evalcls.Integrator import Integrator

from ..utils import evutils

from ..lib import analytics as alyt

from .. import constants

class McolPEvaluator(Integrator):
    """ Coulomb matrix element in momentum space.

        Compute Coulomb matrix element in momentum space by convolving
        Coulomb wave with plane wave matrix element.
        
        Attributes:
            psiColP: Coulomb wave function in momentum space.
            MP: plane wave matrix element.
            area: see sumrules::evalcls::Integrator.
            absErr: see sumrules::evalcls::Integrator.
            relErr: see sumrules::evalcls::Integrator.
    """

    def __init__(self, mp, psicolp):
        """ Init.
            
            By default integration is conducted over a ball of radius `500`.
        """
        super(McolPEvaluator, self).__init__()
        self._keylist += ["maxP"
                         ,"absErr"
                         ,"relErr"]

        self.area = sp.array(((0, 500), (0, sp.pi), (0, 2*sp.pi)))
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
        """ Compute McolP value.
            
            Args:
                p: out-momentum.
                q: in-momentum.
                Tpq: azimuthal angle of out- state relatively to in-state.
                Fpq: polar angle of out- state relatively to in-state.

            Returns:
                Complex float, value of Coulomb matrix element.
        """
        area = self.areaCyclics()

        res, err = cubature(self.McolP_f\
                           ,area.shape[0], 2\
                           ,area.T[0]\
                           ,area.T[1]\
                           ,args=(p, q, Tpq, Fpq)\
                           ,abserr=self.absErr, relerr=self.relErr\
                           ,vectorized=self.vectorized)

        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, p, q, Tpq, Fpq):
        """ Subintegral function.
            
            Args:
                x_args: 3D momentum of an out- plane wave.
                    * x_args[0] for it's radial component.
                    * x_args[1] for it's azimuthal component.
                    * x_args[2] for it's polar component.
                p: momentum of out-state
                q: momentum of in-state
                Tpq: azimuthal angle of out-state relatively to in-state.
                Fpq: polar angle of out-state relatively to in-state.

            Returns:
                NumPy array for `cubature`. The value of subintegral func.
        """
        x_args = self.xargsCyclics(x_args)

        res = self.cubMap(lambda px:\
                    sp.sin(px[1])\
                    *px[0]**2/(2*sp.pi)**3*(alyt.energ(p, self.CONST["m"])/alyt.energ(px[0], self.CONST["m"]))\
                    *sp.conj(self.psiColP(p, px[0], sp.arccos(alyt.coAngle(sp.cos(Tpq), sp.cos(px[1]), px[2]-Fpq))))\
                    *self.MP(px[0], q, px[1], px[2])\
                ,x_args)
        res = self.cyclicPrefactor()*sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(evutil.stackArgRes(x_args, res, sp.array((p, q, Tpq))))
        
        return res


class McolPDiscEvaluator(Integrator):
    """ Coulomb matrix element for discrete spectrum.
        
        Attributes:
            psiColP: Coulomb wave function.
            denerg: discrete energy levels corresponding to `psiColP`.
            MP: plane wave matrix element.
            area: see sumrules::evalcls::Integrator.
            absErr: see sumrules::evalcls::Integrator.
            relErr: see sumrules::evalcls::Integrator.
    """

    def __init__(self, mp, psicolp, denerg):
        super(McolPDiscEvaluator, self).__init__()
        self._keylist += ["maxP"
                         ,"absErr"
                         ,"relErr"]

        self.psiColP = psicolp
        self.denerg = denerg
        self.MP = mp
        
        self.area = sp.array(((0, 500), (0, sp.pi), (0, 2*sp.pi)))
        # be careful with precision, I spent some time to find out its importance here
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
        """ Compute Coulomb matrix element.
            
            Args:
                n: energy level. Main quantum number.
                l: orbital quantum number.

            Returns:
                Float. Value of the matrix element.
        """
        area = self.areaCyclics()

        res, err = cubature(self.McolP_f\
                           ,area.shape[0], 2\
                           ,area.T[0]\
                           ,area.T[1]\
                           ,args=(n, l)\
                           ,abserr=self.absErr, relerr=self.relErr\
                           ,vectorized=self.vectorized)
        
        return res[0] + 1j*res[1]

    def McolP_f(self, x_args, n, l):
        """ Subintegral function.

            Args:
                x_args: 3D intermediate momentum.
                    * x_args[0] it's radial component.
                    * x_args[1] it's azimuthal component.
                    * x_args[2] it's polar component.
                n: energy level. Main quantum number.
                l: orbiral quantum number.

            Returns:
                NumPy array for `cubature`. The value of subintegral func.
        """
        x_args = self.xargsCyclics(x_args)

        res = self.cyclicPrefactor()*self.cubMap(lambda px:\
                    sp.sin(px[1])\
                    *px[0]**2/(2*sp.pi)**3\
                    *sp.sqrt(self.denerg(n, l)/2)/alyt.energ(px[0], self.CONST["m"])\
                    *self.psiColP(n, l\
                                 ,px[0], px[1], px[2])\
                    *sp.conj(\
                        self.MP(px[0]\
                               ,alyt.mom(self.denerg(n, l)**2)\
                               ,px[1], px[2])\
                    )\
                , x_args)
        res = self.cyclicPrefactor()*sp.array((sp.real(res), sp.imag(res))).T

        if self.monitor is not None:
            self.monitor.push(evutils.stackArgRes(x_args, res, sp.array((n,l))))
        
        return res


class GammaDiscEvaluator(Evaluator):
    """ Decay width for discrete spectrum.

        Attributes:
            MPEvaluatorInstance: instance of sumrules::evalcls::Evaluator.
                Should `compute` matrix element corresponding to the process.
            denerg: discrete energy levels of decaying system.
                By default is taken from `MPEvaluatorInstance.denerg`.
    """

    def __init__(self, mp):
        """ Init.
            
            Args:
                mp: instance of sumrules::evalcls::Evaluator.
                    Should `compute` matrix element corresponding to the process.
        """
        super(GammaDiscEvaluator, self).__init__()
        self.MPEvaluatorInstance = mp
        self.denerg = self.MPEvaluatorInstance.denerg

    def compute(self, n, l):
        """ Compute decay width.
            
            Args:
                n: energy level. Main quantum number.
                l: orbital quantum number.

            Returns:
                Float. Value of decay width.
        """
        res = self.CONST["dimfactor"]*self.CONST["Nc"]/(2*l+1)\
                /(16*sp.pi*self.denerg(n, l))\
                *sp.absolute(self.MPEvaluatorInstance.compute(n, l))**2

        if self.monitor is not None:
            self.monitor.push(evutils.stackArgRes(sp.array((n, l)), res))

        return res


class SigmaEvaluator(Integrator):
    """ Evaluate a cross section of a process in c.o.m.
        
        Attributes:
            MPEvaluatorInstance: instance of sumrules::evalcls::Evaluator.
                Should `compute` matrix element of the process.
            area: see sumrules::evalcls::Integrator.
            absErr: see sumrules::evalcls::Integrator.
            relErr: see sumrules::evalcls::Integrator.
    """
    
    def __init__(self, mp):
        """ Init.

            Args:
                mp: instance of sumrules::evalcls::Evaluator.
                    Should `compute` matrix element of the process.
        """
        super(SigmaEvaluator, self).__init__()
        self._keylist += ["absErr"
                         ,"relErr"]

        self.area = sp.array(((0, sp.pi), (0, 2*sp.pi)))
        self.MPEvaluatorInstance = mp

        self.absErr = 1e-5
        self.relErr = 1e-3

    def compute(self, s):
        """ Compute cross-section.
            
            Args:
                s: Mandelstam variable of the process.

            Returns:
                Float. Value of cross-section.
        """
        area = self.areaCyclics()

        res, err = cubature(self.sigma_f\
                           ,area.shape[0], 1\
                           ,area.T[0]\
                           ,area.T[1]\
                           ,args=(s,)\
                           ,abserr=self.absErr, relerr=self.relErr\
                           ,vectorized=self.vectorized)
        
        return res[0]

    def sigma_f(self, x_args, s):
        """ Subintegral function.
            
            Args:
                x_args: solid angle.
                    * x_args[0] it's azimuthal component.
                    * x_args[1] it's polar component.
                s: Mandelstam variable of corresponding process.

            Returns:
                NumPy array for `cubature`. The value of subintegral func.
        """
        x_args = self.xargsCyclics(x_args)

        # use dimfactor for absErr to be reasonable.
        res = self.cubMap(lambda px:\
                sp.sin(px[0])\
                *self.CONST["dimfactor"]*self.CONST["Nc"]*alyt.beta(s)/64/sp.pi**2/s\
                *sp.absolute(\
                             self.MPEvaluatorInstance.compute(alyt.mom(s, self.CONST["m"])\
                                                             ,alyt.mom(s)\
                                                             ,px[0]\
                                                             ,px[1])\
                            )**2\
                , x_args)
        res *= self.cyclicPrefactor()

        if self.monitor is not None:
            self.monitor.push(evutils.stackArgRes(x_args, res, sp.array((s,))))

        return res


class SumruleEvaluator(Integrator):
    """ Evaluator for sumrule in continuous spectrum.

        Attributes:
            SigmaEvaluatorInstance: instance of sumrules::evalcls::Evaluator.
                Should `compute` cross-section corresponding to process.
            area: see sumrules::evalcls::Integrator.
            absErr: see sumrules::evalcls::Integrator.
            relErr: see sumrules::evalcls::Integrator.
    
    """

    def __init__(self, sigma):
        """ Init.
            
            Args:
                sigma: instance of sumrules::evalcls::Evaluator to
                    fill `sigmaEvaluatorInstance`.

                    Should `compute` cross-section of corresponding process.
        """

        super(SumruleEvaluator, self).__init__()
        self._keylist += ["absErr"
                         ,"relErr"
                         ,"minS"
                         ,"maxS"]

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
        """ Compute sumrule.
        
            Returns:
                Float. Value of sumrule.
        """
        area = self.areaCyclics()
        res, err = cubature(self.sumrule_f\
                           ,area.shape[0], 1\
                           ,area.T[0]\
                           ,area.T[1]\
                           ,abserr=self.absErr, relerr=self.relErr\
                           ,vectorized=self.vectorized)
        return res[0]

    def sumrule_f(self, x_args):
        """ Subintegral function.
            
            Args:
                x_args: variable to integrate over.
                    * x_args[0] an `s` Mandelstam variable.

            Returns:
                NumPy array for `cubature`. The value of subintegral func.
        """
        x_args = self.xargsCyclics(x_args)

        res = self.cubMap(lambda px:\
                self.SigmaEvaluatorInstance.compute(px[0])/px[0], x_args)
        res *= self.cyclicPrefactor()

        if self.monitor is not None:
            self.monitor.push(evutils.stackArgRes(x_args, res))

        return res


class SumruleDiscEvaluator(Evaluator):
    """ Evaluate sumrule for discrete spectrum.
        
        Attributes:
            GammaEvaluatorInstance: instance of sumrules::evalcls::Evaluator.
                Should `compute` a decay rate of the process.
            denerg: energy levels of corresponding decaying particle.
            nMax: number of energy levels to take into account while
                computing sum over discrete levels.
    """

    def __init__(self, gamma):
        """ Init.
            
            Args:
                gamma: instance of sumrules::evalcls::Evaluator.
                    Needed to fill `GammaEvaluatorInstance` attribute.
        """
        super(SumruleDiscEvaluator, self).__init__()
        self._keylist += ["nMax"]

        self.GammaEvaluatorInstance = gamma
        self.denerg = self.GammaEvaluatorInstance.denerg

        self.nMax = 10

    def compute(self):
        """ Compute sumrule.
            
            Returns:
                Float. Value of sumrule.
        """
        res = sum(self.mapper(\
                      lambda qn: 16*sp.pi**2*(2*qn[1]+1)\
                                /self.denerg(*qn)**3\
                                *self.GammaEvaluatorInstance.compute(*qn)\
                     ,sp.array( [(n, l)\
                            for n in range(1, self.nMax+1)\
                            for l in range(n)
                      ])
                ))

        return res


class TrivialEvaluator(Evaluator):
    """ Converts analytical function to Evaluator.
        
        Create instance of sumrules::evalcls::Evaluator by providing
        Pythonic function. You can define one by yourself or choose
        from sumrules::lib::analytics.

        Attributes:
            func: the function.
    """

    def __init__(self, func):
        """ Init.
            
            Args:
                func: the function to be converted.
        """
        super(TrivialEvaluator, self).__init__()
        self.func = func

    def compute(self, *args, **kwargs):
        """ Compute `func`.
            
            Args:
                *args: positional args of `func`.
                **kwargs: keyword args of `func`.

            Returns:
                Computed value of `func` on `*args` and `**kwargs`.
        """
        res = self.func(*args, **kwargs)

        if self.monitor is not None:
            adaptedArgs = sp.column_stack(sp.broadcast(*args)).T
            self.monitor.push(evutils.stackArgRes(adaptedArgs, res))

        return res

