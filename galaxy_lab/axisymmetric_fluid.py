"""Finite-domain axisymmetric isothermal Jeans--Poisson equilibrium.

The halo responds to the actual flattened baryonic potential. An even Legendre
expansion and the isolated Newtonian Green function solve halo self-gravity.
The density is cut at a declared spherical computational radius; its nonzero
edge pressure requires imposed boundary stresses. This is a local interior
equilibrium, NOT a globally isolated finite isothermal galaxy or a SIDM
cross-section/viscosity model. Domain and multipole refinement are mandatory.
No observed rotation-curve derivatives enter this forward calculation.

The same static equations apply to an isotropic collisionless isothermal
distribution. References: https://arxiv.org/abs/1311.6524 and
https://arxiv.org/abs/2511.10765 (including its boundary-matching caveat).
Units: kpc, Msun, km/s, and (km/s)^2 for potential.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from numpy.polynomial.legendre import leggauss, legval, legvander
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import CubicSpline
from .fluid import G, IsothermalFluidHalo, _positive_scalar


@dataclass(frozen=True)
class MiyamotoNagai:
    """Positive finite-mass flattened baryons; a=0 gives a Plummer sphere.

    In the equatorial plane, the force depends on a+b, not a and b separately.
    Plane rotation templates therefore cannot determine disk thickness.
    """
    mass_msun: float
    a_kpc: float
    b_kpc: float

    def __post_init__(self):
        if not np.isscalar(self.mass_msun) or not np.isfinite(self.mass_msun) or self.mass_msun < 0:
            raise ValueError("mass_msun must be a finite nonnegative scalar")
        if not np.isscalar(self.a_kpc) or not np.isfinite(self.a_kpc) or self.a_kpc < 0:
            raise ValueError("a_kpc must be a finite nonnegative scalar")
        _positive_scalar(self.b_kpc, "b_kpc")

    def potential(self, R, z):
        return -G*self.mass_msun/np.sqrt(np.asarray(R)**2+(self.a_kpc+np.hypot(z, self.b_kpc))**2)

    def gradient(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        h = np.hypot(z, self.b_kpc)
        den = (R*R+(self.a_kpc+h)**2)**1.5
        return G*self.mass_msun*R/den, G*self.mass_msun*(self.a_kpc+h)*z/(h*den)

    def density(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        h = np.hypot(z, self.b_kpc)
        a, b, m = self.a_kpc, self.b_kpc, self.mass_msun
        return m*b*b/(4*np.pi)*(a*R*R+(a+3*h)*(a+h)**2)/(h**3*(R*R+(a+h)**2)**2.5)

    def circular_velocity_squared(self, R):
        R = np.asarray(R, float)
        return G*self.mass_msun*R*R/(R*R+(self.a_kpc+self.b_kpc)**2)**1.5


def _green_coefficients(r, density_coefficients, orders, central_density):
    """Independent interior/exterior radial integrals of each density multipole."""
    phi, radial_grad, inner_mass = [], [], []
    for j, ell in enumerate(orders):
        rho = density_coefficients[:, j]
        inner = cumulative_trapezoid(rho*r**(ell+2), r, initial=0.)
        if ell == 0:
            inner += central_density*r[0]**3/3
        outer = -cumulative_trapezoid((rho*r**(1-ell))[::-1], r[::-1], initial=0.)[::-1]
        factor = 4*np.pi*G/(2*ell+1)
        value = -factor*(inner/r**(ell+1)+r**ell*outer)
        if ell == 0:
            # Phi_0(r)-Phi_0(0), avoiding cancellation of a large central offset.
            i1 = cumulative_trapezoid(rho*r, r, initial=0.)+central_density*r[0]**2/2
            value = factor*(i1-inner/r)
        phi.append(value)
        radial_grad.append(factor*((ell+1)*inner/r**(ell+2)-ell*r**(ell-1)*outer))
        inner_mass.append(inner[-1])
    phi = np.stack(phi, axis=1)
    grad = np.stack(radial_grad, axis=1)
    central_phi = -4*np.pi*G*(trapezoid(density_coefficients[:, 0]*r, r)+central_density*r[0]**2/2)
    return phi, grad, np.asarray(inner_mass), central_phi


@dataclass(frozen=True)
class AxisymmetricFluidHalo:
    rho0_msun_kpc3: float
    sigma_kms: float
    baryons: tuple[MiyamotoNagai, ...]
    outer_radius_kpc: float = 100.
    radial_points: int = 320
    lmax: int = 12
    angular_points: int = 40
    mixing: float = .25
    tolerance: float = 2e-7
    max_iterations: int = 500

    def __post_init__(self):
        _positive_scalar(self.rho0_msun_kpc3, "rho0_msun_kpc3")
        _positive_scalar(self.sigma_kms, "sigma_kms")
        _positive_scalar(self.outer_radius_kpc, "outer_radius_kpc")
        if self.radial_points < 80 or self.lmax < 0 or self.lmax % 2 or self.angular_points < self.lmax+4:
            raise ValueError("Require >=80 radial points, nonnegative even lmax, and angular_points>=lmax+4")
        if not 0 < self.mixing <= 1 or self.tolerance <= 0 or self.max_iterations < 1:
            raise ValueError("Invalid iteration controls")
        object.__setattr__(self, "baryons", tuple(self.baryons))
        if any(not isinstance(b, MiyamotoNagai) for b in self.baryons):
            raise TypeError("Every baryon component must be MiyamotoNagai")

    def baryon_potential(self, R, z):
        result = np.zeros(np.broadcast_shapes(np.shape(R), np.shape(z)))
        for b in self.baryons:
            result = result+b.potential(R, z)
        return result

    @cached_property
    def _solution(self):
        scale = self.sigma_kms/np.sqrt(4*np.pi*G*self.rho0_msun_kpc3)
        minimum_scale = min([scale, self.outer_radius_kpc]+[b.b_kpc for b in self.baryons])
        r = np.geomspace(minimum_scale*1e-4, self.outer_radius_kpc, self.radial_points)
        mu, weights = leggauss(self.angular_points)
        orders = np.arange(0, self.lmax+1, 2)
        polynomials = legvander(mu, self.lmax)[:, orders]
        projection = weights[:, None]*polynomials*(2*orders+1)/2
        R, z = r[:, None]*np.sqrt(1-mu*mu), r[:, None]*mu
        db = self.baryon_potential(R, z)-self.baryon_potential(0., 0.)
        alone = IsothermalFluidHalo(self.rho0_msun_kpc3, self.sigma_kms)
        log_density = np.log(alone.density(r))[:, None]-db/self.sigma_kms**2
        history = []
        converged = False
        for iteration in range(self.max_iterations):
            density = np.exp(log_density)
            coefficients = density@projection
            # Numerical angular quadrature must not amplify pure roundoff at the origin.
            tiny = np.abs(coefficients[:, 1:]) < coefficients[:, :1]*1e-12
            coefficients[:, 1:][tiny] = 0.
            phi, grad, moments, central_phi = _green_coefficients(r, coefficients, orders, self.rho0_msun_kpc3)
            halo_offset = phi@polynomials.T
            new_log = np.log(self.rho0_msun_kpc3)-(halo_offset+db)/self.sigma_kms**2
            residual = float(np.max(abs(new_log-log_density)))
            history.append(residual)
            if not np.isfinite(residual):
                raise RuntimeError("Nonfinite self-consistent axisymmetric iteration")
            if residual < self.tolerance:
                converged = True
                break
            log_density = (1-self.mixing)*log_density+self.mixing*new_log
        if not converged:
            raise RuntimeError(f"Axisymmetric equilibrium did not converge: log-density residual={residual:.4g}")
        # Spline coefficients represent halo potential offset; every ell>0 is
        # unchanged by the central reference. Boundary gravity is its complete
        # finite-domain even-multipole vacuum Green solution.
        spline = CubicSpline(np.log(r), phi, axis=0)
        return dict(r=r, mu=mu, orders=orders, coefficients=coefficients,
                    phi=phi, radial_gradient=grad, moments=moments,
                    central_phi=central_phi, spline=spline,
                    iterations=iteration+1, residual=residual, history=history,
                    sampled_density=density)

    @property
    def diagnostics(self):
        s = self._solution
        return dict(converged=True, iterations=s["iterations"], fixed_point_log_density_residual=s["residual"],
                    outer_radius_kpc=self.outer_radius_kpc, radial_points=self.radial_points,
                    lmax=self.lmax, angular_points=self.angular_points,
                    boundary="Finite-density halo cut at sphere; isolated exterior halo multipoles; imposed confining stresses required",
                    total_halo_mass_inside_domain_msun=float(4*np.pi*s["moments"][0]),
                    maximum_boundary_pressure=float(np.max(s["sampled_density"][-1])*self.sigma_kms**2),
                    minimum_boundary_pressure=float(np.min(s["sampled_density"][-1])*self.sigma_kms**2))

    def _coordinates(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        if not np.isfinite(R).all() or not np.isfinite(z).all() or np.any(R < 0):
            raise ValueError("R must be nonnegative and R,z finite")
        radius = np.hypot(R, z)
        if np.any(radius > self.outer_radius_kpc*(1+1e-13)):
            raise ValueError("Evaluation outside finite computational domain is not allowed")
        mu = np.divide(z, radius, out=np.zeros_like(radius), where=radius > 0)
        return R, z, radius, mu

    def halo_potential_offset(self, R, z):
        _, _, r, mu = self._coordinates(R, z)
        s = self._solution
        bounded = np.maximum(r, s["r"][0])
        values = s["spline"](np.log(bounded))
        small = r < s["r"][0]
        # Smooth central potential: leading l0 and l2 terms are O(r²).
        if np.any(small):
            factors = (r[small]/s["r"][0])[:, None]**np.maximum(s["orders"], 2)
            values[small] *= factors
        basis = legvander(mu, self.lmax)[..., s["orders"]]
        return np.sum(values*basis, axis=-1)

    def potential_offset(self, R, z):
        return self.halo_potential_offset(R, z)+self.baryon_potential(R, z)-self.baryon_potential(0., 0.)

    def density(self, R, z):
        return self.rho0_msun_kpc3*np.exp(-self.potential_offset(R, z)/self.sigma_kms**2)

    def pressure(self, R, z):
        return self.sigma_kms**2*self.density(R, z)

    def equatorial_circular_velocity(self, R):
        R = np.asarray(R, float)
        _, _, _, _ = self._coordinates(R, np.zeros_like(R))
        s = self._solution
        positive = R > 0
        derivative = np.zeros_like(R)
        rr = np.maximum(R[positive], s["r"][0])
        basis = legvander(np.zeros_like(rr), self.lmax)[..., s["orders"]]
        derivative[positive] = np.sum(s["spline"](np.log(rr), 1)*basis, axis=-1)/rr
        tiny = positive & (R < s["r"][0])
        if np.any(tiny):
            derivative[tiny] *= R[tiny]/s["r"][0]
        for b in self.baryons:
            derivative += b.gradient(R, np.zeros_like(R))[0]
        v2 = R*derivative
        if np.any(v2 < -1e-7):
            raise RuntimeError("Negative total equatorial circular force")
        return np.sqrt(np.maximum(v2, 0.))

    def poisson_density(self, R, z):
        """Halo density from an independent differential Laplacian of potential.

        Useful for closure diagnostics away from the origin and boundary.
        This differentiates a potential spline, not the fitted observations.
        """
        _, _, r, mu = self._coordinates(R, z)
        s = self._solution
        if np.any(r <= s["r"][0]):
            raise ValueError("Poisson diagnostic requires radius above inner integration start")
        t = np.log(r)
        ell = s["orders"]
        coefficients = (s["spline"](t, 2)+s["spline"](t, 1)-ell*(ell+1)*s["spline"](t))/r[..., None]**2
        return np.sum(coefficients*legvander(mu, self.lmax)[..., ell], axis=-1)/(4*np.pi*G)

    def exterior_halo_potential(self, radius, mu):
        """Vacuum continuation of the finite-domain halo potential, zero at infinity.

        Baryonic gravity is separate. This validates the gravitational boundary,
        not mechanical isolation: the fluid still has nonzero pressure at its cut.
        """
        radius, mu = np.broadcast_arrays(np.asarray(radius, float), np.asarray(mu, float))
        if not np.isfinite(radius).all() or not np.isfinite(mu).all() or np.any(radius < self.outer_radius_kpc) or np.any(abs(mu) > 1):
            raise ValueError("Require finite exterior radius>=outer_radius and |mu|<=1")
        s = self._solution
        ell = s["orders"]
        coeff = -4*np.pi*G/(2*ell+1)*s["moments"]/radius[...,None]**(ell+1)
        return np.sum(coeff*legvander(mu,self.lmax)[...,ell],axis=-1)
