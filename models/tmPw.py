import scipy as sp
from scipy import special
from cubature import cubature

from .basic import config as bconfig
m = bconfig["m"]
dimfactor = bconfig["dimfactor"]
from .basic import mom, beta, eta, coAngle

from ..config import config as mconfig

# Matrix elements
def MX(r, q, Crq):
    return sp.exp(-m*r)/4/sp.pi/r*(sp.exp(1j*r*q*Crq) + sp.exp(-1j*r*q*Crq))

def MP(q, p, Cqp):
    return 1/(q**2 + p**2 - 2*p*q*Cqp + m**2) + 1/(q**2 + p**2 + 2*q*p*Cqp + m**2)

def sigma_f(x_args, p):
    """
        x_args:
            px[0] - Cpq
        p:
            s, MP
    """
    px = x_args.T

    return dimfactor*beta(p["s"])/32/sp.pi/p["s"]*(sp.absolute(p["MP"](mom(p["s"]), mom(p["s"], m), px[0]))**2).T

def sigma(p):
    """
        p:
            s, MP
    """

    res, err = cubature(sigma_f, 1, 1, [-1], [1], args=[p], abserr=mconfig["abs_err"], relerr=mconfig["rel_err"], vectorized=True)
    return res[0]
