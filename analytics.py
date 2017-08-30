import scipy as sp
from scipy import special

from . import constants as CONST

# Kinematics
def mom(s, m=0):
    return sp.sqrt(s/4 - m**2)

def energ(p, m=0):
    return sp.sqrt(p**2 + m**2)

def beta(s):
    return sp.sqrt(1 - 4*CONST["m"]**2/s)

def eta(k):
    return CONST["mu"]*CONST["g"]/k

def coAngle(Cpq, Cpr, Fqr):
    return Cpq*Cpr + sp.sqrt(1 - Cpq**2)*sp.sqrt(1 - Cpr**2)*sp.cos(Fqr)

# Matrix elements
def tmMX(r, q, Crq):
    return sp.exp(-CONST["m"]*r)/4/sp.pi/r*(sp.exp(1j*r*q*Crq) + sp.exp(-1j*r*q*Crq))

def tmMP(q, p, Cqp, Fpq):
    return 1/(q**2 + p**2 - 2*p*q*Cqp + CONST["m"]**2) + 1/(q**2 + p**2 + 2*q*p*Cqp + CONST["m"]**2)

def sqedMP0(p, q, Cpq, Fpq):
    return 2j*CONST["e1"]**2*(1 - p**2*(1-Cpq**2)*(1/(p**2 + q**2 - 2*p*q*Cpq + CONST["m"]**2) + 1/(p**2 + q**2 + 2*p*q*Cpq + CONST["m"]**2)))

def sqedMP2(p, q, Cpq, Fpq):
    return 2j*CONST["e1"]**2*p**2*(1-Cpq**2)*(1/(p**2 + q**2 - 2*p*q*Cpq + CONST["m"]**2) + 1/(p**2 + q**2 + 2*p*q*Cpq + CONST["m"]**2))*sp.exp(2j*Fpq)

# Wave functions
def psiColP(k, p, Ckp):
    "\psi^{col}_p(k, p, Ckp) - Coulomb wave in momentum space"
    return -4*sp.pi*sp.exp(-sp.pi*eta(k)/2)*sp.special.gamma(1 + 1j*eta(k))*(
            2*(p**2 - (k + 1j*CONST["eps"])**2)**(1j*eta(k))*CONST["eps"]*(-1 - 1j*eta(k))/(p**2 + k**2 - 2*p*k*Ckp + CONST["eps"]**2)**(2 + 1j*eta(k)) +
            2*(k +  1j*CONST["eps"])*eta(k)*(p**2 - (k + 1j*CONST["eps"])**2)**(-1 + 1j*eta(k))/(p**2 + k**2 - 2*p*k*Ckp + CONST["eps"]**2)**(1 + 1j*eta(k))
            )
