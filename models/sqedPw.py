import scipy as sp
from scipy import special
from cubature import cubature

from multiprocessing.dummy import Pool as ThPool

from .basic import config as bconfig
from .basic import mom, energ, beta, eta, coAngle
from ..config import config

# Translate params dict to globals
m = bconfig["m"]
e1 = bconfig["e1"]
dimfactor = bconfig["dimfactor"]

# Matrix elements
def MP0(p, q, Cpq, Fpq):
    return 2j*e1**2*(1 - p**2*(1-Cpq**2)*(1/(p**2 + q**2 - 2*p*q*Cpq + m**2) + 1/(p**2 + q**2 + 2*p*q*Cpq + m**2)))

def MP2(p, q, Cpq, Fpq):
    return 2j*e1**2*p**2*(1-Cpq**2)*(1/(p**2 + q**2 - 2*p*q*Cpq + m**2) + 1/(p**2 + q**2 + 2*p*q*Cpq + m**2))*sp.exp(2j*Fpq)


def sigma_f(x_args, p):
    """
        x_args:
            px[0] - Cpq
        p:
            s, MP
    """
    px = x_args.T

    return dimfactor*beta(p["s"])/32/sp.pi/p["s"]*(sp.absolute(p["MP"](mom(p["s"], m), mom(p["s"]), px[0], 0))**2).T

def sigma(p):
    """
        p:
            s, MP
    """

    res, err = cubature(sigma_f, 1, 1, [-1], [1], args=[p], abserr=config["abs_err"], relerr=config["rel_err"], vectorized=True)
    return res[0]

def sumrule_f(x_args, p):
    """
        x_args:
            px[0] - s

        p:
            MP
    """
    px = x_args.T

    pool = ThPool(config["num_threads"])
    sumrule_evaled = pool.map(lambda s: sigma({"s": s, "MP": p["MP"]})/s, px[0])
    sumrule_evaled = sp.array(sumrule_evaled)

    return sumrule_evaled.T


def sumrule(p):
    """
        p:
            MP, minS, maxS
    """

    res, err = cubature(sumrule_f, 1, 1, [p["minS"]], [p["maxS"]], args=[p], abserr=config["abs_err"], relerr=config["rel_err"], vectorized=True)
    return res[0]
