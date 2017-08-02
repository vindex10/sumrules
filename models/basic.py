import scipy as sp
from scipy import special

# Parameters

config = {"g": 0.6 # "+" for repulsive, "-" for attractive
         ,"m": 1.27
         ,"e1": 0.303
         ,"eps": 0.01
         ,"dimfactor": 38940}
config.update({"mu": config["m"]/2})

# Kinematics

def mom(s, m=0):
    return sp.sqrt(s/4 - config["m"]**2)

def beta(s):
    return sp.sqrt(1 - 4*config["m"]**2/s)

def eta(k):
    return config["mu"]*config["g"]/k

def coAngle(Cpq, Cpr, Fqr):
    return Cpq*Cpr + sp.sqrt(1 - Cpq**2)*sp.sqrt(1 - Cpr**2)*sp.cos(Fqr)
