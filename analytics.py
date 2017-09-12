import scipy as sp
from scipy import special
from scipy import misc

from . import constants as CONST

# Kinematics
def mom(s, m=0):
    return sp.sqrt(s/4 - m**2)

def energ(p, m=0):
    return sp.sqrt(p**2 + m**2)

def beta(s):
    return sp.sqrt(1 - 4*CONST["m"]**2/s)

def eta(k):
    return CONST["m"]*CONST["g"]/2/k

def coAngle(Cpq, Cpr, Fqr):
    return Cpq*Cpr + sp.sqrt(1 - Cpq**2)*sp.sqrt(1 - Cpr**2)*sp.cos(Fqr)

# Matrix elements
def tmMX(r, q, Trq):
    return sp.exp(-CONST["m"]*r)/4/sp.pi/r\
            *(sp.exp(1j*r*q*sp.cos(Trq)) + sp.exp(-1j*r*q*sp.cos(Trq)))

def tmMP(q, p, Tqp, Fpq):
    return 1/(q**2 + p**2 - 2*p*q*sp.cos(Tqp) + CONST["m"]**2)\
                +\
           1/(q**2 + p**2 + 2*q*p*sp.cos(Tqp) + CONST["m"]**2)

def sqedMP0(p, q, Tpq, Fpq):
    return 2j*CONST["e"]**2*(1 - p**2*sp.sin(Tpq)**2\
            *(\
                1/(p**2 + q**2 - 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
                    +\
                1/(p**2 + q**2 + 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
             ))

def sqedMP2(p, q, Tpq, Fpq):
    return 2j*CONST["e"]**2*p**2*sp.sin(Tpq)**2\
            *(\
                1/(p**2 + q**2 - 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
                    +\
                1/(p**2 + q**2 + 2*p*q*sp.cos(Tpq) + CONST["m"]**2))*sp.exp(2j*Fpq\
            )

# Wave functions
def psiColP(k, p, Tkp):
    "\psi^{col}_p(k, p, Ckp) - Coulomb wave in momentum space"
    return -4*sp.pi*sp.exp(-sp.pi*eta(k)/2)*sp.special.gamma(1 + 1j*eta(k))\
            *(\
                2*(p**2 - (k + 1j*CONST["eps"])**2)**(1j*eta(k))*CONST["eps"]\
                *(-1 - 1j*eta(k))/(p**2 + k**2 - 2*p*k*sp.cos(Tkp) + CONST["eps"]**2)**(2 + 1j*eta(k))\
                    +\
                2*(k +  1j*CONST["eps"])*eta(k)*(p**2 - (k + 1j*CONST["eps"])**2)**(-1 + 1j*eta(k))\
                /(p**2 + k**2 - 2*p*k*sp.cos(Tkp) + CONST["eps"]**2)**(1 + 1j*eta(k))\
            )

def psiColPdisc(n, l, M, p, Tpq, Fpq):
    if M not in range(-l, l+1):
        return 0
    if l not in range(n):
        return 0
    return (-1j)**l *(4*sp.pi) *2**(2*(l+1)) *sp.misc.factorial(l)/(n+l)**2\
            *sp.sqrt(sp.special.factorial(n-1)/sp.special.factorial(n+2*l))\
            *(2/CONST["m"]/(-CONST["g"]))**(3/2)*((p*(2/CONST["m"]/(-CONST["g"])))/(n+l))**l\
            /(((2/CONST["m"]/(-CONST["g"]))*p)**2 + 1/(n+l)**2)**(2+l)\
            *sp.special.eval_gegenbauer(n-1\
                                      , l+1\
                                      , ((n+l)**2*(p*(2/CONST["m"]/(-CONST["g"])))**2 - 1)\
                                        /((n+l)**2*(p*(2/CONST["m"]/(-CONST["g"])))**2 + 1))\
            *sp.special.sph_harm(M, l, Fpq, Tpq)

# Spectra
def energColDisc(n, l):
    return 2*CONST["m"] - CONST["m"]*CONST["g"]**2/4/(n+l)**2
