"""A pressure-supported dark-halo baseline with explicit astrophysical units.

This replaces the draft's stellar drag balance; it does not repair that balance.
The spherical, nonrotating dark component obeys ``P = rho * sigma**2``,
``dP/dr = -rho * G*M/r**2``, and ``dM/dr = 4*pi*r**2*rho``.  Tracer stars
respond to its gravity, with circular speed ``sqrt(G*M/r)``.  A collisionless
isotropic isothermal distribution has the same equilibrium equations: fitting
this model therefore cannot establish fluid microphysics or viscosity.

The untruncated isothermal sphere has infinite total mass.  Use it as a local
rotation-curve baseline, not a complete cosmological halo.  Adding a disk's
rotation contribution afterward is an *uncontracted halo approximation*, not
an exact hydrostatic solution in the combined disk and halo potential.

References (primary theory, not claims of a new discovery):
  https://arxiv.org/abs/1311.6524
  https://academic.oup.com/mnras/article/521/3/4630/7074577
  https://farside.ph.utexas.edu/teaching/336L/Fluid/node20.html
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

G = 4.30091727003628e-6  # kpc (km/s)^2 / Msun
KPC_M = 3.085677581491367e19
MSUN_KG = 1.988409870698051e30
GYR_S = 3.15576e16
SHEAR_UNIT_PER_S = 1000.0 / KPC_M  # (km/s)/kpc -> 1/s
_X_MIN = 1.0e-3
_X_MAX = 1.0e8


def _finish(value, original):
    """Preserve scalar inputs while allowing NumPy broadcasting."""
    result = np.asarray(value)
    return float(result) if np.ndim(original) == 0 and result.ndim == 0 else result


def _finite_nonnegative(value, name):
    result = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _core_series(x):
    """Regular Lane-Emden series for psi and w=x*dpsi/dx at the origin."""
    x2 = x * x
    psi = x2 * (1.0 / 6.0 + x2 * (-1.0 / 120.0 + x2 / 1890.0))
    w = x2 * (1.0 / 3.0 + x2 * (-1.0 / 30.0 + x2 / 315.0))
    return psi, w


@lru_cache(maxsize=1)
def _universal_solution():
    """Tabulate once in log radius; interpolation respects dpsi/dlog(x)=w.

    With t=log(x), dpsi/dt=w and dw/dt=exp(2*t-psi)-w.  This avoids the
    singular coefficient at x=0 and works across many decades in radius.
    A regular series handles x<=1e-3; no singular-isothermal approximation is
    substituted for the numerically integrated outer solution.
    """
    t = np.linspace(np.log(_X_MIN), np.log(_X_MAX), 8193)
    initial = np.asarray(_core_series(np.asarray(_X_MIN)))

    def rhs(log_x, state):
        psi, w = state
        return (w, np.exp(2.0 * log_x - psi) - w)

    solution = solve_ivp(
        rhs, (t[0], t[-1]), initial, method="DOP853", t_eval=t,
        rtol=2.0e-12, atol=2.0e-15,
    )
    if not solution.success:
        raise RuntimeError(f"Lane-Emden integration failed: {solution.message}")
    return CubicHermiteSpline(t, solution.y[0], solution.y[1], extrapolate=False)


def dimensionless_profile(x):
    """Return ``(psi, w)`` for x=r/r0, with rho/rho0=exp(-psi), v²/sigma²=w.

    Supports finite ``0 <= x <= 1e8``.  The origin is exact: ``psi=w=0``.
    ``w*x`` is the dimensionless enclosed mass.  Negative or out-of-range
    radii raise an error instead of silently extrapolating.
    """
    radius = _finite_nonnegative(x, "dimensionless radius x")
    if np.any(radius > _X_MAX):
        raise ValueError("dimensionless radius x exceeds tabulation limit 1e8")
    psi = np.empty_like(radius)
    w = np.empty_like(radius)
    small = radius <= _X_MIN
    psi[small], w[small] = _core_series(radius[small])
    if np.any(~small):
        spline = _universal_solution()
        log_x = np.log(radius[~small])
        psi[~small] = spline(log_x)
        w[~small] = spline(log_x, 1)
    return _finish(psi, x), _finish(w, x)


@dataclass(frozen=True)
class IsothermalFluidHalo:
    """Regular, self-gravitating spherical isothermal halo.

    Parameters
    ----------
    rho0_msun_kpc3 : float
        Central dark-matter density in solar masses per cubic kiloparsec.
    sigma_kms : float
        Constant one-dimensional velocity dispersion (or isothermal sound
        speed) in km/s.  The asymptotic circular speed is sqrt(2)*sigma.

    The scale r0=sigma/sqrt(4*pi*G*rho0) is sometimes called a core radius.
    It is explicitly *not* a half-density radius; conventions for core radius
    differ.  This class uses r0 as its declared core-radius convention.
    """

    rho0_msun_kpc3: float
    sigma_kms: float

    def __post_init__(self):
        _positive_scalar(self.rho0_msun_kpc3, "rho0_msun_kpc3")
        _positive_scalar(self.sigma_kms, "sigma_kms")

    @property
    def scale_radius_kpc(self):
        return self.sigma_kms / np.sqrt(4.0 * np.pi * G * self.rho0_msun_kpc3)

    @property
    def core_radius_kpc(self):
        """Declared r0 convention, equal to scale_radius_kpc."""
        return self.scale_radius_kpc

    @property
    def asymptotic_speed_kms(self):
        return np.sqrt(2.0) * self.sigma_kms

    def _profile(self, r_kpc):
        radius = _finite_nonnegative(r_kpc, "r_kpc")
        psi, w = dimensionless_profile(radius / self.scale_radius_kpc)
        return radius, np.asarray(psi), np.asarray(w)

    def density(self, r_kpc):
        """Dark-matter mass density in Msun/kpc³."""
        _, psi, _ = self._profile(r_kpc)
        return _finish(self.rho0_msun_kpc3 * np.exp(-psi), r_kpc)

    def enclosed_mass(self, r_kpc):
        """Mass enclosed within r in Msun, exactly zero at r=0."""
        radius, _, w = self._profile(r_kpc)
        return _finish(self.sigma_kms**2 * radius * w / G, r_kpc)

    def circular_velocity(self, r_kpc):
        """Circular gravitational tracer speed in km/s, zero at r=0."""
        _, _, w = self._profile(r_kpc)
        return _finish(self.sigma_kms * np.sqrt(np.maximum(w, 0.0)), r_kpc)

    def pressure(self, r_kpc):
        """Effective pressure rho*sigma² in Msun (km/s)²/kpc³.

        This may represent isotropic velocity-dispersion stress.  It is not a
        measured thermodynamic temperature of dark matter.
        """
        return self.density(r_kpc) * self.sigma_kms**2

    def potential_offset(self, r_kpc):
        """Phi(r)-Phi(0) in (km/s)².  Infinity is not a finite reference."""
        _, psi, _ = self._profile(r_kpc)
        return _finish(self.sigma_kms**2 * psi, r_kpc)

    def dlog_density_dr(self, r_kpc):
        """Analytic dln(rho)/dr in 1/kpc, including its regular zero limit."""
        radius, _, w = self._profile(r_kpc)
        derivative = np.divide(-w, radius, out=np.zeros_like(radius), where=radius > 0)
        return _finish(derivative, r_kpc)


def isothermal_velocity(r_kpc, rho0_msun_kpc3, sigma_kms):
    """Physical-parameter convenience wrapper; returns circular speed in km/s."""
    return IsothermalFluidHalo(rho0_msun_kpc3, sigma_kms).circular_velocity(r_kpc)


def fluid_circular_speed(r_kpc, log10_rho0_Msun_kpc3, log10_sigma_kms):
    """Optimizer API: log10 physical density and dispersion, speed in km/s."""
    if not np.isfinite(log10_rho0_Msun_kpc3) or not np.isfinite(log10_sigma_kms):
        raise ValueError("log parameters must be finite")
    return isothermal_velocity(r_kpc, 10.0**log10_rho0_Msun_kpc3, 10.0**log10_sigma_kms)


def cylindrical_shear(r_kpc, v_phi_kms, dv_dr_kms_per_kpc):
    """Axisymmetric rotational shear ``dv_phi/dr - v_phi/r`` in (km/s)/kpc.

    Supply a derivative of a forward velocity model when doing inference.
    At r=0 the v/r limit requires a model, so only strictly positive r is
    accepted.  Rigid rotation has zero shear; a flat curve has -v/r shear.
    This is a fluid's velocity field, not automatically the stellar curve.
    """
    radius = _finite_nonnegative(r_kpc, "r_kpc")
    velocity = np.asarray(v_phi_kms, dtype=float)
    derivative = np.asarray(dv_dr_kms_per_kpc, dtype=float)
    if np.any(radius == 0):
        raise ValueError("cylindrical shear requires strictly positive r_kpc")
    if np.any(~np.isfinite(velocity)) or np.any(~np.isfinite(derivative)):
        raise ValueError("velocity and derivative must be finite")
    return _finish(derivative - velocity / radius, r_kpc)


def viscous_dissipation_w_m3(shear_kms_per_kpc, dynamic_viscosity_pa_s):
    """Pure-rotation Newtonian heating ``mu * shear²`` in W/m³.

    Dynamic viscosity mu must be >=0, ensuring nonnegative heating.  This is
    a diagnostic for a separately specified flowing fluid.  The equilibrium
    halo above is nonrotating and has no such shear or viscous dissipation.
    """
    shear = np.asarray(shear_kms_per_kpc, dtype=float)
    viscosity = _finite_nonnegative(dynamic_viscosity_pa_s, "dynamic_viscosity_pa_s")
    if np.any(~np.isfinite(shear)):
        raise ValueError("shear must be finite")
    return _finish(viscosity * (shear * SHEAR_UNIT_PER_S) ** 2, shear_kms_per_kpc)


def kinematic_viscosity_m2_s(dynamic_viscosity_pa_s, density_msun_kpc3):
    """Convert nu=mu/rho to m²/s using mu in Pa s and rho in Msun/kpc³."""
    viscosity = _finite_nonnegative(dynamic_viscosity_pa_s, "dynamic_viscosity_pa_s")
    density = _finite_nonnegative(density_msun_kpc3, "density_msun_kpc3")
    if np.any(density == 0):
        raise ValueError("density_msun_kpc3 must be strictly positive")
    return _finish(viscosity / (density * MSUN_KG / KPC_M**3), dynamic_viscosity_pa_s)


def viscous_timescale_gyr(length_kpc, kinematic_viscosity_m2_s):
    """Order-of-magnitude diffusion time L²/nu in Gyr.

    This dimensional timescale omits boundary/geometry factors; it does not
    fit or infer viscosity from a static rotation curve.  At nu=0, a positive
    length has infinite diffusion time; L=nu=0 is undefined and rejected.
    """
    length, viscosity = np.broadcast_arrays(
        _finite_nonnegative(length_kpc, "length_kpc"),
        _finite_nonnegative(kinematic_viscosity_m2_s, "kinematic_viscosity_m2_s"),
    )
    if np.any((length == 0) & (viscosity == 0)):
        raise ValueError("L=nu=0 gives an undefined diffusion time")
    result = np.divide((length * KPC_M) ** 2, viscosity,
                       out=np.full_like(length, np.inf), where=viscosity > 0) / GYR_S
    return _finish(result, length_kpc)
