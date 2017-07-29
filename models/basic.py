import scipy as sp
from scipy import special

# Parameters

g = -0.6
m = 1.27
e1 = 0.303
mu = m/2
eps = 0.01
INF = 100


# Kinematics

def mom(s, m=0):
    return sp.sqrt(s/4 - m**2)

def beta(s):
    return sp.sqrt(1 - 4*m**2/s)

def eta(k):
    return mu*g/k

def coAngle(Cpq, Cpr, Fqr):
    return Cpq*Cpr + sp.sqrt(1 - Cpq**2)*sp.sqrt(1 - Cpr**2)*sp.cos(Fqr)
