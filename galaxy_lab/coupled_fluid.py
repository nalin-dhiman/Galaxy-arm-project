"""Spherical isothermal Jeans/pressure equilibrium in baryon + halo gravity.

Unlike adding a disk force to a previously isolated halo, this module solves
the dark density in the *total spherical potential*. It is exact within its
stated assumptions: spherical baryons, nonrotating isotropic isothermal dark
component, static equilibrium. It is not an axisymmetric galaxy-disk solver.
No viscosity, scattering cross section, distance, or evolutionary stability is
inferred by these static equations. The untruncated halo has infinite mass.

Equations: rho_dm = rho0 exp[-(Phi_dm-Phi_dm0+Phi_b-Phi_b0)/sigma²],
dM_dm/dr = 4 pi r² rho_dm. Primary references:
https://arxiv.org/abs/1311.6524
https://academic.oup.com/mnras/article/521/3/4630/7074577
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from .fluid import G, _finite_nonnegative, _finish, _positive_scalar


class SphericalBaryons(Protocol):
    """A positive spherical mass profile with finite central potential.

    All methods use kpc, Msun and km/s. ``central_density_slope`` is
    dln(rho_dm)/dr at r=0 for an isothermal tracer with the supplied sigma.
    This protocol is implemented by PlummerBaryons and HernquistBaryons;
    equatorial disk forces must not be passed as enclosed spherical mass.
    """
    mass_msun: float
    scale_kpc: float

    def enclosed_mass(self, r_kpc): ...
    def density(self, r_kpc): ...
    def potential_offset(self, r_kpc): ...
    def central_density_slope(self, sigma_kms): ...


@dataclass(frozen=True)
class PlummerBaryons:
    mass_msun: float
    scale_kpc: float

    def __post_init__(self):
        _finite_nonnegative(self.mass_msun, "mass_msun")
        if not np.isscalar(self.mass_msun):
            raise ValueError("mass_msun must be a scalar")
        _positive_scalar(self.scale_kpc, "scale_kpc")

    def enclosed_mass(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        return _finish(self.mass_msun*r**3/(r*r+self.scale_kpc**2)**1.5, r_kpc)

    def density(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        return _finish(3*self.mass_msun/(4*np.pi*self.scale_kpc**3)
                       *(1+(r/self.scale_kpc)**2)**-2.5, r_kpc)

    def potential_offset(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        # Stable at r << a: 1-1/sqrt(1+x²) = x²/[s(s+1)].
        x = r/self.scale_kpc
        s = np.hypot(1., x)
        return _finish(G*self.mass_msun/self.scale_kpc*x*x/(s*(s+1)), r_kpc)

    def central_density_slope(self, sigma_kms):
        return 0.


@dataclass(frozen=True)
class HernquistBaryons:
    """Finite-mass cusped baryons, rho=M a/[2 pi r(r+a)^3].

    The baryon density diverges at r=0 while the potential remains finite.
    Halo density remains finite; its right-hand central derivative is nonzero.
    """
    mass_msun: float
    scale_kpc: float

    def __post_init__(self):
        _finite_nonnegative(self.mass_msun, "mass_msun")
        if not np.isscalar(self.mass_msun):
            raise ValueError("mass_msun must be a scalar")
        _positive_scalar(self.scale_kpc, "scale_kpc")

    def enclosed_mass(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        return _finish(self.mass_msun*(r/(r+self.scale_kpc))**2, r_kpc)

    def density(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        if self.mass_msun == 0:
            return _finish(np.zeros_like(r), r_kpc)
        with np.errstate(divide="ignore"):
            result = self.mass_msun*self.scale_kpc/(2*np.pi*r*(r+self.scale_kpc)**3)
        return _finish(result, r_kpc)

    def potential_offset(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        return _finish(G*self.mass_msun/self.scale_kpc*r/(r+self.scale_kpc), r_kpc)

    def central_density_slope(self, sigma_kms):
        return -G*self.mass_msun/(self.scale_kpc**2*sigma_kms**2)


@dataclass(frozen=True)
class CoupledSphericalFluidHalo:
    rho0_msun_kpc3: float
    sigma_kms: float
    baryons: SphericalBaryons
    max_radius_kpc: float = 300.
    rtol: float = 2.e-10

    def __post_init__(self):
        _positive_scalar(self.rho0_msun_kpc3, "rho0_msun_kpc3")
        _positive_scalar(self.sigma_kms, "sigma_kms")
        _positive_scalar(self.max_radius_kpc, "max_radius_kpc")
        _positive_scalar(self.rtol, "rtol")
        if not isinstance(self.baryons, (PlummerBaryons, HernquistBaryons)):
            raise TypeError("baryons must be a validated PlummerBaryons or HernquistBaryons profile")

    @property
    def scale_radius_kpc(self):
        """The dark-only regular-sphere scale; not the coupled core radius."""
        return self.sigma_kms/np.sqrt(4*np.pi*G*self.rho0_msun_kpc3)

    @cached_property
    def _start_radius(self):
        depth = G*self.baryons.mass_msun/(self.baryons.scale_kpc*self.sigma_kms**2)
        return min(self.scale_radius_kpc, self.baryons.scale_kpc,
                   self.max_radius_kpc)*1.e-6/max(1., depth)

    def _central_dm(self, r):
        # Integral of rho=rho0*(1+slope*r+...) supplies a regular start,
        # including the Hernquist baryon cusp's finite central acceleration.
        a = 4*np.pi*G*self.rho0_msun_kpc3/self.sigma_kms**2
        slope = self.baryons.central_density_slope(self.sigma_kms)
        return a*(r*r/6+slope*r**3/12), a*(r*r/3+slope*r**3/4)

    @cached_property
    def _solution(self):
        r0 = self._start_radius
        def rhs(log_r, y):
            r = np.exp(log_r)
            total_psi = y[0]+self.baryons.potential_offset(r)/self.sigma_kms**2
            source = 4*np.pi*G*self.rho0_msun_kpc3*r*r*np.exp(-total_psi)/self.sigma_kms**2
            return (y[1], source-y[1])
        result = solve_ivp(rhs, (np.log(r0), np.log(self.max_radius_kpc)),
                           self._central_dm(r0), method="DOP853", dense_output=True,
                           rtol=self.rtol, atol=1.e-25)
        if not result.success:
            raise RuntimeError(f"Coupled Jeans-Poisson integration failed: {result.message}")
        return result.sol

    def _profile(self, r_kpc):
        r = _finite_nonnegative(r_kpc, "r_kpc")
        if np.any(r > self.max_radius_kpc):
            raise ValueError("r_kpc exceeds max_radius_kpc; no extrapolation is allowed")
        psi_dm, w_dm = (np.asarray(value) for value in self._central_dm(r))
        outer = r > self._start_radius
        if np.any(outer):
            psi_dm[outer], w_dm[outer] = self._solution(np.log(r[outer]))
        return r, psi_dm, w_dm

    def density(self, r_kpc):
        r, psi_dm, _ = self._profile(r_kpc)
        psi_total = psi_dm+self.baryons.potential_offset(r)/self.sigma_kms**2
        return _finish(self.rho0_msun_kpc3*np.exp(-psi_total), r_kpc)

    def enclosed_mass(self, r_kpc):
        """Dark mass only, in Msun; total_mass includes the baryons."""
        r, _, w_dm = self._profile(r_kpc)
        return _finish(self.sigma_kms**2*r*w_dm/G, r_kpc)

    def total_mass(self, r_kpc):
        return self.enclosed_mass(r_kpc)+self.baryons.enclosed_mass(r_kpc)

    def circular_velocity(self, r_kpc):
        """Tracer speed due to the *combined spherical* baryons + dark halo."""
        r, _, _ = self._profile(r_kpc)
        v2 = np.divide(G*self.total_mass(r), r, out=np.zeros_like(r), where=r > 0)
        return _finish(np.sqrt(np.maximum(v2, 0)), r_kpc)

    def pressure(self, r_kpc):
        return self.density(r_kpc)*self.sigma_kms**2

    def potential_offset(self, r_kpc):
        """Total potential Phi(r)-Phi(0) in (km/s)²."""
        r, psi_dm, _ = self._profile(r_kpc)
        return _finish(self.sigma_kms**2*psi_dm+self.baryons.potential_offset(r), r_kpc)

    def dlog_density_dr(self, r_kpc):
        r, _, _ = self._profile(r_kpc)
        result = np.divide(-G*self.total_mass(r), self.sigma_kms**2*r*r,
                           out=np.full_like(r, self.baryons.central_density_slope(self.sigma_kms)),
                           where=r > 0)
        return _finish(result, r_kpc)
