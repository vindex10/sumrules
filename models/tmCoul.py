from __future__ import absolute_import, division
from builtins import * # quite boldly but simply enough

import scipy as sp
from scipy import special
from multiprocessing.dummy import Pool as ThPool
from cubature import cubature

from .basic import config as bconfig
m = bconfig["m"]
eps = bconfig["eps"]
dimfactor = bconfig["dimfactor"]
from .basic import mom, beta, eta, coAngle

from ..config import config as pconfig

config = {"maxP": 100}

# Wave functions

def psiColP(k, p, Ckp):
    "\psi^{col}_p(k, p, Ckp) - Coulomb wave in momentum space"
    return -4*sp.pi*sp.exp(-sp.pi*eta(k)/2)*sp.special.gamma(1 + 1j*eta(k))*(
            2*(p**2 - (k + 1j*eps)**2)**(1j*eta(k))*eps*(-1 - 1j*eta(k))/(p**2 + k**2 - 2*p*k*Ckp + eps**2)**(2 + 1j*eta(k)) +
            2*(k +  1j*eps)*eta(k)*(p**2 - (k + 1j*eps)**2)**(-1 + 1j*eta(k))/(p**2 + k**2 - 2*p*k*Ckp + eps**2)**(1 + 1j*eta(k))
            )


# Matrix elements
def MX(r, q, Crq):
    return sp.exp(-m*r)/4/sp.pi/r*(sp.exp(1j*r*q*Crq) + sp.exp(-1j*r*q*Crq))

def MP(q, p, Cqp):
    return 1/(q**2 + p**2 - 2*p*q*Cqp + m**2) + 1/(q**2 + p**2 + 2*q*p*Cqp + m**2)

def McolP_f(x_args, p):
    """
        x_args:
            px[0] for px
            px[1] for Cos(px,p)
            px[2] for phi
        p (for params):
            p, q, Cpq, psiColP, MP
    """
    px = x_args.T

    res = px[0]**2/(2*sp.pi)**3*sp.sqrt((p["p"]**2+m**2)/(px[0]**2+m**2))*sp.conj(psiColP(p["p"], px[0], coAngle(p["Cpq"], px[1], px[2])))*p["MP"](px[0], p["q"], px[1])
    return sp.vstack((sp.real(res), sp.imag(res))).T

def McolP(p):
    """
        p (for params):
            p, q, Cpq, psiColP, MP
    """
    res, err = cubature(McolP_f, 3, 2, [0, -1, 0], [config["maxP"], 1, 2*sp.pi], args=[p], abserr=pconfig["abs_err"], relerr=pconfig["rel_err"], vectorized=True)
    return res[0] + 1j*res[1]

def sigma_f(x_args, p):
    """
        x_args:
            px[0] - Cpq
        p:
            s, psiColP, MP
    """
    px = x_args.T

    #Multiprocessing to eval McolP for multiple args in parallel
    pool = ThPool(pconfig["num_threads"])
    McolP_evaled = pool.map(lambda x: sp.absolute(McolP({"p": mom(p["s"], m), "q": mom(p["s"]), "Cpq": x, "psiColP": p["psiColP"], "MP": p["MP"]}))**2, px[0])
    McolP_evaled = sp.array(McolP_evaled)

    return dimfactor*beta(p["s"])/32/sp.pi/p["s"]*McolP_evaled.T

def sigma(p):
    """
        p:
            s, psiColP, MP
    """

    res, err = cubature(sigma_f, 1, 1, [-1], [1], args=[p], abserr=pconfig["abs_err"], relerr=pconfig["rel_err"], vectorized=True)
    return res[0]
