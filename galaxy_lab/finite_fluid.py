"""Globally finite lowered-isothermal equilibrium in a flattened potential.

The isotropic nonnegative distribution f(E)=A[exp((E_t-E)/s²)-1] for E<E_t,
zero otherwise, supplies both density and pressure. Its pressure is NOT rho*s²:
s is a velocity scale and the actual local one-dimensional dispersion is P/rho.
This is an ergodic collisionless equilibrium, equivalently a static barotropic
Jeans/fluid closure. It neither assumes nor identifies dark-matter viscosity.

Halo self-gravity uses an isolated Newtonian Green function. The computational
sphere must enclose the entire zero-density equipotential; otherwise the solve
raises rather than hiding a pressure discontinuity. Baryonic MN components are
finite mass but need not have compact support. The energy cutoff is an explicit
model assumption, not an inferred cosmological/tidal history.

References: King1966 doi:10.1086/109857; Gieles & Zocchi2015 arXiv:1508.02120.
Units: kpc, Msun, km/s. No observed velocity derivatives are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import erf

from .fluid import G, _positive_scalar
from .axisymmetric_fluid import AxisymmetricFluidHalo, MiyamotoNagai, _green_coefficients


def king_moment(W, pressure=False):
    """Dimensionless density or pressure integral, stable at the zero edge.

    Density starts as W**(5/2), pressure as W**(7/2); dK_pressure/dW=K_density.
    Values at nonpositive relative binding are exactly zero.
    """
    raw=np.asarray(W,float)
    w=np.maximum(raw,0.)
    if not np.isfinite(raw).all() or np.any(w>100):
        raise ValueError('Finite King binding W<=100 required')
    small=w<.5
    result=np.zeros_like(w)
    if np.any(small):
        x=w[small]
        term=(16/(105*np.sqrt(np.pi))*x**3.5 if pressure
              else 8/(15*np.sqrt(np.pi))*x**2.5)
        total=term.copy()
        power=3.5 if pressure else 2.5
        for _ in range(18):
            term=term*x/(power+1)
            total+=term
            power+=1
        result[small]=total
    if np.any(~small):
        x=w[~small]
        poly=1+2*x/3+(4*x*x/15 if pressure else 0)
        result[~small]=np.exp(x)*erf(np.sqrt(x))-2*np.sqrt(x/np.pi)*poly
    return result


@lru_cache(maxsize=32)
def _spherical_dimensionless(W0):
    """Independent spherical King ODE with first zero as a terminating event."""
    normalization=float(king_moment(W0))
    x0=1e-6
    def rhs(x,y):
        return [-y[1]/(x*x), x*x*float(king_moment(y[0]))/normalization]
    def edge(x,y):return y[0]
    edge.terminal=True
    edge.direction=-1
    sol=solve_ivp(rhs,(x0,1e6),[W0-x0*x0/6,x0**3/3],events=edge,
                  rtol=2e-9,atol=1e-11,dense_output=True,max_step=2.)
    if not sol.success or not len(sol.t_events[0]):
        raise RuntimeError('Spherical lowered-isothermal reference has no resolved finite edge')
    return sol,float(sol.t_events[0][0]),float(sol.y_events[0][0,1])


@dataclass(frozen=True)
class SphericalFiniteHalo:
    """Baryon-free ODE reference and globally finite spherical baseline."""
    rho0_msun_kpc3: float
    sigma_kms: float
    central_binding: float=5.

    def __post_init__(self):
        _positive_scalar(self.rho0_msun_kpc3,'rho0_msun_kpc3')
        _positive_scalar(self.sigma_kms,'sigma_kms')
        if not .1<=self.central_binding<=15:
            raise ValueError('central_binding must lie in [.1,15]')

    @property
    def scale_kpc(self):return self.sigma_kms/np.sqrt(4*np.pi*G*self.rho0_msun_kpc3)

    @property
    def edge_radius_kpc(self):return _spherical_dimensionless(float(self.central_binding))[1]*self.scale_kpc

    @property
    def total_mass_msun(self):return 4*np.pi*self.rho0_msun_kpc3*self.scale_kpc**3*_spherical_dimensionless(float(self.central_binding))[2]

    def _coordinates(self,r):
        r=np.asarray(r,float)
        if not np.isfinite(r).all() or np.any(r<0):raise ValueError('Finite nonnegative radius required')
        return r,r/self.scale_kpc

    def relative_binding(self,r):
        r,x=self._coordinates(r)
        sol,edge,mass=_spherical_dimensionless(float(self.central_binding))
        out=np.empty_like(x)
        inside=x<=edge
        if np.any(inside):out[inside]=sol.sol(np.maximum(x[inside],1e-6))[0]
        if np.any(~inside):out[~inside]=mass*(1/x[~inside]-1/edge)
        out[x==0]=self.central_binding
        return out

    def density(self,r):return self.rho0_msun_kpc3*king_moment(self.relative_binding(r))/king_moment(self.central_binding)

    def pressure(self,r):return self.rho0_msun_kpc3*self.sigma_kms**2*king_moment(self.relative_binding(r),True)/king_moment(self.central_binding)

    def circular_velocity(self,r):
        r,x=self._coordinates(r)
        sol,edge,total=_spherical_dimensionless(float(self.central_binding))
        mass=np.full_like(x,total)
        inner=x<=edge
        if np.any(inner):mass[inner]=sol.sol(np.maximum(x[inner],1e-6))[1]
        value=np.divide(mass,x,out=np.zeros_like(x),where=x>0)
        return self.sigma_kms*np.sqrt(np.maximum(value,0.))


@dataclass(frozen=True)
class FiniteFluidHalo(AxisymmetricFluidHalo):
    """Self-consistent flattened finite King halo with zero-pressure free edge.

    outer_radius_kpc=None starts at three times the baryon-free King edge and
    enlarges the sphere if necessary. Explicit
    domains can be used to verify vacuum-domain independence. Enclosure is
    checked densely at the sphere; truncating nonzero density is forbidden.
    """
    outer_radius_kpc: float | None=None
    central_binding: float=5.

    def __post_init__(self):
        if not .1<=self.central_binding<=15:
            raise ValueError('central_binding must lie in [.1,15]')
        automatic=self.outer_radius_kpc is None
        object.__setattr__(self,'_automatic_domain',automatic)
        if automatic:
            reference=SphericalFiniteHalo(self.rho0_msun_kpc3,self.sigma_kms,self.central_binding)
            object.__setattr__(self,'outer_radius_kpc',3.*reference.edge_radius_kpc)
        super().__post_init__()

    @cached_property
    def _solution(self):
        domain=self.outer_radius_kpc
        for attempt in range(7 if self._automatic_domain else 1):
            try:
                result=self._solve_domain(domain)
                result['domain_enlargements']=attempt
                return result
            except RuntimeError as exc:
                if 'does not enclose' not in str(exc) or not self._automatic_domain:raise
                domain*=3
        raise RuntimeError('Automatic domain did not enclose the free surface after seven enlargements')

    def _solve_domain(self,domain):
        reference=SphericalFiniteHalo(self.rho0_msun_kpc3,self.sigma_kms,self.central_binding)
        minimum=min([reference.scale_kpc,domain]+[b.b_kpc for b in self.baryons])
        r=np.geomspace(minimum*1e-4,domain,self.radial_points)
        mu,weights=leggauss(self.angular_points)
        orders=np.arange(0,self.lmax+1,2)
        basis=legvander(mu,self.lmax)[:,orders]
        projection=weights[:,None]*basis*(2*orders+1)/2
        R,z=r[:,None]*np.sqrt(1-mu*mu),r[:,None]*mu
        db=self.baryon_potential(R,z)-self.baryon_potential(0.,0.)
        norm=float(king_moment(self.central_binding))
        density=np.broadcast_to(reference.density(r)[:,None],R.shape).copy()
        history=[];previous_binding=None
        for iteration in range(self.max_iterations):
            coefficients=density@projection
            tiny=np.abs(coefficients[:,1:])<coefficients[:,:1]*1e-12
            coefficients[:,1:][tiny]=0.
            phi,grad,moments,central_phi=_green_coefficients(r,coefficients,orders,self.rho0_msun_kpc3)
            binding=self.central_binding-(phi@basis.T+db)/self.sigma_kms**2
            new_density=self.rho0_msun_kpc3*king_moment(binding)/norm
            # Absolute central-density normalized convergence remains defined
            # where the physical free surface gives exactly zero density.
            residual=float(np.max(abs(new_density-density))/self.rho0_msun_kpc3)
            binding_change=(float(np.max(abs(binding-previous_binding)))
                            if previous_binding is not None else np.inf)
            history.append(residual)
            if not np.isfinite(residual):raise RuntimeError('Nonfinite finite-halo iteration')
            if residual<self.tolerance and binding_change<self.tolerance*5:break
            density=(1-self.mixing)*density+self.mixing*new_density
            previous_binding=binding
        else:
            raise RuntimeError(f'Finite halo did not converge: density residual={residual:g}')
        spline=CubicSpline(np.log(r),phi,axis=0)
        # Include polar/equatorial extrema; Gauss nodes alone miss the poles.
        boundary_mu=np.linspace(-1,1,401)
        boundary_halo=legvander(boundary_mu,self.lmax)[:,orders]@phi[-1]
        boundary_db=self.baryon_potential(r[-1]*np.sqrt(1-boundary_mu**2),r[-1]*boundary_mu)-self.baryon_potential(0.,0.)
        maximum_boundary_binding=float(np.max(self.central_binding-(boundary_halo+boundary_db)/self.sigma_kms**2))
        if maximum_boundary_binding>=-1e-5:
            raise RuntimeError('Computational domain does not enclose zero-pressure surface; increase outer_radius_kpc')
        cutoff=float(central_phi+self.baryon_potential(0.,0.)+self.sigma_kms**2*self.central_binding)
        if cutoff>=0:raise RuntimeError('Nonnegative cutoff energy would not define isolated finite support')
        return dict(r=r,mu=mu,orders=orders,coefficients=coefficients,phi=phi,
                    radial_gradient=grad,moments=moments,central_phi=central_phi,
                    spline=spline,iterations=iteration+1,residual=residual,history=history,
                    sampled_density=density,maximum_boundary_binding=maximum_boundary_binding,
                    cutoff_energy_kms2=cutoff,binding_iteration_change=binding_change)

    @property
    def diagnostics(self):
        s=self._solution
        return dict(converged=True,iterations=s['iterations'],
                    fixed_point_density_residual_over_rho0=s['residual'],
                    outer_radius_kpc=float(s['r'][-1]),radial_points=self.radial_points,
                    automatic_domain_enlargements=s['domain_enlargements'],
                    binding_iteration_change=s['binding_iteration_change'],
                    lmax=self.lmax,angular_points=self.angular_points,central_binding=self.central_binding,
                    maximum_boundary_binding=s['maximum_boundary_binding'],
                    cutoff_energy_kms2=s['cutoff_energy_kms2'],
                    total_halo_mass_msun=float(4*np.pi*s['moments'][0]),
                    maximum_boundary_pressure=0.,
                    closure='Lowered-isothermal ergodic distribution; variable P/rho; no viscosity identified',
                    boundary='Zero-density zero-pressure equipotential enclosed inside numerical sphere; vacuum Green continuation')

    def _coordinates(self,R,z):
        R,z=np.broadcast_arrays(np.asarray(R,float),np.asarray(z,float))
        if not np.isfinite(R).all() or not np.isfinite(z).all() or np.any(R<0):
            raise ValueError('R must be nonnegative and coordinates finite')
        radius=np.hypot(R,z)
        if np.any(radius>self._solution['r'][-1]*(1+1e-13)):
            raise ValueError('Use exterior_halo_potential outside the solved vacuum domain')
        mu=np.divide(z,radius,out=np.zeros_like(radius),where=radius>0)
        return R,z,radius,mu

    def exterior_halo_potential(self,radius,mu):
        radius,mu=np.broadcast_arrays(np.asarray(radius,float),np.asarray(mu,float))
        s=self._solution
        if (not np.isfinite(radius).all() or not np.isfinite(mu).all()
            or np.any(radius<s['r'][-1]) or np.any(abs(mu)>1)):
            raise ValueError('Require exterior radius at or beyond actual solved domain and |mu|<=1')
        ell=s['orders']
        coeff=-4*np.pi*G/(2*ell+1)*s['moments']/radius[...,None]**(ell+1)
        return np.sum(coeff*legvander(mu,self.lmax)[...,ell],axis=-1)

    def halo_potential_offset(self,R,z):
        R,z=np.broadcast_arrays(np.asarray(R,float),np.asarray(z,float))
        if not np.isfinite(R).all() or not np.isfinite(z).all() or np.any(R<0):
            raise ValueError('Finite coordinates and R>=0 required')
        radius=np.hypot(R,z);inside=radius<=self._solution['r'][-1]
        result=np.empty_like(radius)
        if np.any(inside):
            result[inside]=super().halo_potential_offset(R[inside],z[inside])
        if np.any(~inside):
            result[~inside]=(self.exterior_halo_potential(radius[~inside],z[~inside]/radius[~inside])
                             -self._solution['central_phi'])
        return result

    def equatorial_circular_velocity(self,R):
        R=np.asarray(R,float)
        if not np.isfinite(R).all() or np.any(R<0):raise ValueError('Finite R>=0 required')
        s=self._solution;inside=R<=s['r'][-1]
        result=np.empty_like(R)
        if np.any(inside):result[inside]=super().equatorial_circular_velocity(R[inside])
        if np.any(~inside):
            r=R[~inside];ell=s['orders']
            halo_v2=np.sum(4*np.pi*G*(ell+1)/(2*ell+1)*s['moments']
                           /r[:,None]**(ell+1)*legvander(np.zeros_like(r),self.lmax)[:,ell],axis=-1)
            baryon_v2=sum(b.circular_velocity_squared(r) for b in self.baryons)
            result[~inside]=np.sqrt(np.maximum(halo_v2+baryon_v2,0.))
        return result

    def relative_binding(self,R,z):return self.central_binding-self.potential_offset(R,z)/self.sigma_kms**2

    def density(self,R,z):return self.rho0_msun_kpc3*king_moment(self.relative_binding(R,z))/king_moment(self.central_binding)

    def pressure(self,R,z):return self.rho0_msun_kpc3*self.sigma_kms**2*king_moment(self.relative_binding(R,z),True)/king_moment(self.central_binding)

    def local_dispersion_squared(self,R,z):
        rho=self.density(R,z)
        return np.divide(self.pressure(R,z),rho,out=np.zeros_like(rho),where=rho>0)


def main():
    """Inspect supplied physical parameters; this does not infer them from pixels."""
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho0',type=float,default=3e7,help='Central halo density [Msun/kpc^3]')
    parser.add_argument('--velocity-scale',type=float,default=100.,help='DF velocity scale [km/s], not the local dispersion')
    parser.add_argument('--binding',type=float,default=5.,help='Dimensionless central binding W0')
    parser.add_argument('--baryon-mass',type=float,default=1e10,help='Supplied MN baryon mass [Msun]')
    parser.add_argument('--a',type=float,default=2.,help='Supplied MN radial parameter [kpc]')
    parser.add_argument('--b',type=float,default=.4,help='Supplied MN vertical parameter [kpc], not exponential scale height')
    parser.add_argument('--radii',type=float,nargs='+',default=[1.,5.,10.,50.,150.])
    args=parser.parse_args()
    halo=FiniteFluidHalo(args.rho0,args.velocity_scale,(MiyamotoNagai(args.baryon_mass,args.a,args.b),),
                         radial_points=640,lmax=24,angular_points=72,central_binding=args.binding,tolerance=1e-9)
    r=np.asarray(args.radii)
    profile=dict(radius_kpc=r.tolist(),total_circular_speed_kms=halo.equatorial_circular_velocity(r).tolist(),
                 halo_density_msun_kpc3=halo.density(r,0.).tolist(),
                 local_1d_dispersion_kms=np.sqrt(halo.local_dispersion_squared(r,0.)).tolist())
    print(json.dumps(dict(diagnostics=halo.diagnostics,profile=profile,
                         interpretation='Forward calculation from supplied physical parameters; no image-only distance, viscosity or dark-matter interaction rate inferred'),indent=2,allow_nan=False))


if __name__=='__main__':main()
