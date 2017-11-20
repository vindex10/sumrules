"""@file
Stores analytical expressions needed for computation.
(It is a pure Python version, because some functions
fail to run in Cython)
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import scipy as sp
from scipy import special
from scipy import misc

from .. import constants as CONST

# Kinematics
def mom(s, m=0):
    """ Returns momentum of a particle
        
        Get absolute value of momentum of a particle in c.o.m. frame.

        @f[
           p = \sqrt{\frac{s}{4} - m^2}
        @f]

        Args:
            s: mandelstam variable. \f$ s = (p_1 + p_2)^2 = (p_3 + p_4)^2 \f$.

        Kwargs:
            m: mass of the particle.

        Returns:
            Float number, absolute value of momentum of a particle.
    """
    return sp.sqrt(s/4 - m**2)

def energ(p, m=0):
    """ Energy of relativistic particle

        @f[
            E = \sqrt{p^2 + m^2}
        @f]

        Args:
            p: momentum of a particle.
        
        Kwargs:
            m: mass of a particle.
        
        Returns:
            Float number, energy of a relativistic particle.
    """
    return sp.sqrt(p**2 + m**2)

def beta(s):
    """ Inverse velocity of particles in the process.
        
        @f[
            \beta(s) = \sqrt{1 - \frac{4 m^2}{s}} = \frac{2}{v_{cm}}
        @f]

        Args:
            s: mandelstam variable characterizing process. \f$ s = (p_1 + p_2)^2 = (p_3 + p_4)^2 \f$.

        Returns:
            Float number, doubled inverse velocity.

        See Also:
            m: sumrules::constants
    """
    return sp.sqrt(1 - 4*CONST["m"]**2/s)

def eta(k):
    """ Sommerfeld enhancement factor.

        @f[
            \eta = \frac{m g}{2 k} = \frac{\mu g}{k}
        @f]

        Args:
            k: momentum of the particle.

        Returns:
            Float number. Sommerfeld enhancement.

        See Also:
            m: sumrules::constants
            g: sumrules::constants
    """
    return CONST["m"]*CONST["g"]/2/k

def coAngle(Cpq, Cpr, Fqr):
    """ Cosine of angle between two vectors

        Provides cosine of the angle between two vectors by their
        spherical coordinates.

        @f[
            \cos(\theta) = \cos(\theta_1)\cos(\theta_2) + \sin(\theta_1)\sin(\theta_2)\cos(\Delta\phi)
        @f]

        Args:
            Cpq: cosine of the azimuthal component of the first vector.
            Cpr: cosine of the azimuthal component of the second vector.
            Fqr: polar angular distance between two vectors.

        Returns:
            Float number. Cosine of angle between two vectors.
    """
    return Cpq*Cpr + sp.sqrt(1 - Cpq**2)*sp.sqrt(1 - Cpr**2)*sp.cos(Fqr)

# Matrix elements
def tmMX(r, q, Trq):
    """ Toy model plane wave matrix element in coordinate representation.

        @f[
            M(r, q, \theta) = \mathrm{e}^{-\frac{m r}{4 \pi r}} \left(\
                        \mathrm{e}^{\mathrm{i} r q \cos(\theta)} + e^{-\mathrm{i} r q \cos(\theta)}
                    \right)
        @f]

        Args:
            r: absolute value of radius vector of out particle.
            q: absolute value of momentum of in particle.
            Trq: cosine of the angle between in and out vectors.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
    """
    return sp.exp(-CONST["m"]*r)/4/sp.pi/r\
            *(sp.exp(1j*r*q*sp.cos(Trq)) + sp.exp(-1j*r*q*sp.cos(Trq)))

def tmMP(q, p, Tpq, Fpq):
    """ Toy model plane wave matrix element in momentum representation.
        
        @f[
            M(q, p, \theta, \phi) = \frac{1}{q^2 + p^2 + 2pq\cos(\theta) + m^2}\
                    +\
            \frac{1}{q^2 + p^2 - 2pq\cos{\theta} + m^2}
        @f]

        Args:
            q: absolute value of momentum of out particle.
            p: absolute value of momentum of in particle.
            Tpq: azimuthal angular distance between particles.
            Fpq: polar angular distance between particles.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
    """
    return 1/(q**2 + p**2 - 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
                +\
           1/(q**2 + p**2 + 2*q*p*sp.cos(Tpq) + CONST["m"]**2)

def sqedMP0(p, q, Tpq, Fpq):
    """ Scalar QED matrix element for spin-0 process in momentum representation.
        
        @f[
            M(p, q, \theta, \phi) = 2\mathrm{i} e^2 (1 - p^2\sin(\theta)^2) \left(
                \frac{1}{p^2 + q^2 - 2pq\cos(\theta) + m^2}\
                    +\
                \frac{1}{p^2 + q^2 + 2pq\cos(\theta) + m^2}\
            \right)
        @f]

        Args:
            q: absolute value of momentum of out particle.
            p: absolute value of momentum of in particle.
            Tpq: azimuthal angular distance between particles.
            Fpq: polar angular distance between particles.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
            e: sumrules::constants
    """
    return 2j*CONST["e"]**2*(1 - p**2*sp.sin(Tpq)**2\
            *(\
                1/(p**2 + q**2 - 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
                    +\
                1/(p**2 + q**2 + 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
             ))

def sqedMP2(p, q, Tpq, Fpq):
    """ Scalar QED matrix element for spin-2 process in momentum representation.
        
        @f[
            M(p, q, \theta, \phi) = 2\mathrm{i} e^2 p^2\sin(\theta)^2 \left(
                \frac{1}{p^2 + q^2 - 2pq\cos(\theta) + m^2}\
                    +\
                    \frac{1}{p^2 + q^2 + 2pq\cos(\theta) + m^2}\
            \right) \mathrm{e}^{2 \mathrm{i} \phi}
        @f]

        Args:
            q: absolute value of momentum of out particle.
            p: absolute value of momentum of in particle.
            Tpq: azimuthal angular distance between particles.
            Fpq: polar angular distance between particles.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
            e: sumrules::constants
    """
    return 2j*CONST["e"]**2*p**2*sp.sin(Tpq)**2\
            *(\
                1/(p**2 + q**2 - 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
                    +\
                1/(p**2 + q**2 + 2*p*q*sp.cos(Tpq) + CONST["m"]**2)\
            )*sp.exp(2j*Fpq)

def sqedMP0onsh(p, q, Tpq, Fpq):
    """ Scalar QED matrix element for spin-0 process in momentum representation.
        
        This version on the contrary to `sqedMP0` uses "on-shell reduction".

        @f[
            M(p, q, \theta, \phi) = 2\mathrm{i} e^2 (1 - p^2\sin(\theta)^2) \left(
                \frac{1}{2q (-E_p + 2p\cos(\theta))}\
                    +\
                \frac{1}{2q (-E_p - 2p\cos(\theta))}\
            \right)
        @f]

        Args:
            q: absolute value of momentum of out particle.
            p: absolute value of momentum of in particle.
            Tpq: azimuthal angular distance between particles.
            Fpq: polar angular distance between particles.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
            e: sumrules::constants
    """
    return 2j*CONST["e"]**2*(1 - p**2*sp.sin(Tpq)**2\
            *(\
                1/(2*q*( -sp.sqrt(p**2 + CONST["m"]**2) + 2*p*sp.cos(Tpq)))\
                    +\
                1/(2*q*( -sp.sqrt(p**2 + CONST["m"]**2) - 2*p*sp.cos(Tpq)))\
             ))

def sqedMP2onsh(p, q, Tpq, Fpq):
    """ Scalar QED matrix element for spin-2 process in momentum representation.
        
        This version on the contrary to `sqedMP0` uses "on-shell reduction".

        @f[
            M(p, q, \theta, \phi) = 2\mathrm{i} e^2 p^2\sin(\theta)^2 \left(
                \frac{1}{2q (-E_p + 2p\cos(\theta))}\
                    +\
                \frac{1}{2q (-E_p - 2p\cos(\theta))}\
            \right) \mathrm{e}^{2 \mathrm{i} \phi}
        @f]

        Args:
            q: absolute value of momentum of out particle.
            p: absolute value of momentum of in particle.
            Tpq: azimuthal angular distance between particles.
            Fpq: polar angular distance between particles.

        Returns:
            Float number. Value of matrix element between in and out states.

        See Also:
            m: sumrules::constants
            e: sumrules::constants
    """
    return 2j*CONST["e"]**2*p**2*sp.sin(Tpq)**2\
            *(\
                1/(2*q*( -sp.sqrt(p**2 + CONST["m"]**2) + 2*p*sp.cos(Tpq)))\
                    +\
                1/(2*q*( -sp.sqrt(p**2 + CONST["m"]**2) - 2*p*sp.cos(Tpq)))\
            )*sp.exp(2j*Fpq)

# Wave functions
def psiColP(k, p, Tkp):
    """ Coulomb wave function in momentum representation.

        Coulomb wave function in momentum representation, which is regularized
        with `eps` parameter. More detailed about Coulomb wave you can read
        in:
        
        https://dx.doi.org/10.1016/j.cpc.2014.10.002

        @f[
            \psi^{Coul}_k(p) = -4 \pi \mathrm{e}^{-\frac{\pi \eta(k)}{2}} \Gamma(1+\mathrm{i}\eta(k)) \left(\
                \frac{2 (p^2 - (k + \mathrm{i} \varepsilon)^2)^{\mathrm{i} \eta(k)} \varepsilon\
                (-1 - \mathrm{i}\eta(k))}{(p^2 + k^2 - 2pk\cos(\theta) + \varepsilon^2)^{(2 + \mathrm{i} \eta(k)}}\
                    +\
                \frac{2 (k +  \mathrm{i} \varepsilon) \eta(k) (p^2 - (k + \mathrm{i}\varepsilon)^2)^{-1 + \mathrm{i} \eta(k)}}{\
                (p^2 + k^2 - 2pk\cos(\theta) + \varepsilon^2)^{1 + \mathrm{i}\eta(k)}}\
            \right)
        @f]

        Args:
            k: absolute value of momentum of bound state.
            p: absolute value of momentum of plane wave.
            Tkp: angle between bound state and plane wave momenta.

        Returns:
            Float number. Value of wave function.

        See Also:
            eta: ::eta()
            m: sumrules::constants
    """
    return -4*sp.pi*sp.exp(-sp.pi*eta(k)/2)*sp.special.gamma(1 + 1j*eta(k))\
            *(\
                2*(p**2 - (k + 1j*CONST["eps"])**2)**(1j*eta(k))*CONST["eps"]\
                *(-1 - 1j*eta(k))/(p**2 + k**2 - 2*p*k*sp.cos(Tkp) + CONST["eps"]**2)**(2 + 1j*eta(k))\
                    +\
                2*(k +  1j*CONST["eps"])*eta(k)*(p**2 - (k + 1j*CONST["eps"])**2)**(-1 + 1j*eta(k))\
                /(p**2 + k**2 - 2*p*k*sp.cos(Tkp) + CONST["eps"]**2)**(1 + 1j*eta(k))\
            )

def psiColPdisc(n, l, M, p, Tpq, Fpq):
    """ Coulomb wave function for discrete spectrum in momentum representation.

        @f[
            \psi_{n, l, M}(p, \theta, \phi) = (-\mathrm{i})^l (4 \pi) 2^{2 (l+1)} \frac{l!}{(n+l)^2}\
                \sqrt{\frac{(n-1)!}{(n+2l)!}}\
                (-\frac{2}{m g})^{3/2} \frac{(\frac{p (-\frac{2}{m g})}{n+l})^l}{\
                ((-\frac{2}{m g}) p)^2 + (\frac{1}{n+l})^2)^{2+l}}\
                G(n-1, l+1\
                    , \frac{(n+l)^2 (p (-\frac{2}{m g}))^2 - 1}{\
                    (n+l)^2 (p (-\frac{2}{m g}))^2 + 1})\
                Y(M, l, \phi, \theta)
        @f]

        It depends on `M` also, but for specific calculations, usually,
        one should partially apply projection of momentum.

        Args:
            n: energy level.
            l: orbital quantum number.
            M: angular momentum projection.
            p: absolute value of momentum of the particle.
            Tpq: azimuthal direction of the particle.
            Fpq: polar direction of the particle.

        Returns:
            Float number. Value of discrete Coulomb wave function.

        See Also:
            m: sumrules::constants
            g: sumrules::constants
            G: [Gegenbauer polynomials](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gegenbauer.html)
            Y: [Spherical harmonics](https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.special.sph_harm.html)
    """

    for bo in sp.broadcast(n, l, M):
        if bo[2] not in range(-bo[1], bo[1]+1):
            return 0
        if bo[1] not in range(bo[0]):
            return 0

    return (-1j)**l *(4*sp.pi) *2**(2*(l+1)) *sp.misc.factorial(l)/(n+l)**2\
            *sp.sqrt(sp.special.factorial(n-1)/sp.special.factorial(n+2*l))\
            *(2/CONST["m"]/(-CONST["g"]))**(3/2)*((p*(2/CONST["m"]/(-CONST["g"])))/(n+l))**l\
            /(((2/CONST["m"]/(-CONST["g"]))*p)**2 + 1/(n+l)**2)**(2+l)\
            *sp.special.eval_gegenbauer(n-1\
                                       ,l+1\
                                       ,((n+l)**2*(p*(2/CONST["m"]/(-CONST["g"])))**2 - 1)\
                                        /((n+l)**2*(p*(2/CONST["m"]/(-CONST["g"])))**2 + 1))\
            *sp.special.sph_harm(M, l, Fpq, Tpq)

def psiColPdisc0(n, l, p, Tpq, Fpq):
    """ Coulomb wave function for spin-0 bound state.
        Force angular momentum projection to be 0 in `psiColPdisc`.

        Args:
            n: energy level.
            l: orbital quantum number.
            p: absolute value of momentum of the particle.
            Tpq: azimuthal direction of the particle.
            Fpq: polar direction of the particle.

        Returns:
            Float number. Value of discrete Coulomb wave function of spin-0 particle.
    """

    return psiColPdisc(n, l, 0, p, Tpq, Fpq)

def psiColPdisc2(n, l, p, Tpq, Fpq):
    """ Coulomb wave function for spin-2 bound state.
        Force angular momentum projection to be 2 in `psiColPdisc`.

        Args:
            n: energy level.
            l: orbital quantum number.
            p: absolute value of momentum of the particle.
            Tpq: azimuthal direction of the particle.
            Fpq: polar direction of the particle.

        Returns:
            Float number. Value of discrete Coulomb wave function of spin-2 particle.
    """

    return psiColPdisc(n, l, 2, p, Tpq, Fpq)

# Spectra
def energColDisc(n, l):
    """ Discrete energy spectrum of the Coulomb bound states.

        @f[
            E_{n,l} = 2m - \frac{m g^2}{4 (n+l)^2}
        @f]

        Args:
            n: energy level.
            l: orbital quantum number.

        Returns:
            Float number. Energy for specific level.

        See Also:
            m: sumrules::constants
            g: sumrules::constants
    """
    return 2*CONST["m"] - CONST["m"]*CONST["g"]**2/4/(n+l)**2
