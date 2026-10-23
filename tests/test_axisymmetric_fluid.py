"""Forward gravitational closure, symmetry and independently known limits."""
import numpy as np
import pytest
from galaxy_lab.axisymmetric_fluid import AxisymmetricFluidHalo, MiyamotoNagai
from galaxy_lab.fluid import G, IsothermalFluidHalo
from galaxy_lab.coupled_fluid import CoupledSphericalFluidHalo, PlummerBaryons


def test_miyamoto_nagai_positive_density_poisson_and_force_closure():
    b = MiyamotoNagai(4e10, 3., .4)
    R, z = np.array([.2, 1., 3., 10.]), np.array([.1, .4, 1., 4.])
    h = 1e-3
    p = b.potential(R,z)
    dR = (b.potential(R+h,z)-b.potential(R-h,z))/(2*h)
    dz = (b.potential(R,z+h)-b.potential(R,z-h))/(2*h)
    laplace = (b.potential(R+h,z)-2*p+b.potential(R-h,z))/h**2+dR/R+(b.potential(R,z+h)-2*p+b.potential(R,z-h))/h**2
    np.testing.assert_allclose(b.gradient(R,z), np.array([dR,dz]), rtol=4e-6)
    np.testing.assert_allclose(laplace, 4*np.pi*G*b.density(R,z), rtol=5e-5)
    assert np.all(b.density(R,z) > 0)


def test_equatorial_baryon_force_does_not_identify_disk_thickness():
    a = MiyamotoNagai(4e10, 3., .4)
    b = MiyamotoNagai(4e10, 2.4, 1.)
    R = np.geomspace(.01, 30., 30)
    np.testing.assert_allclose(a.circular_velocity_squared(R), b.circular_velocity_squared(R), rtol=1e-14)
    assert not np.isclose(a.potential(2.,2.), b.potential(2.,2.))


def test_zero_baryon_limit_against_independent_spherical_lane_emden():
    h = AxisymmetricFluidHalo(3e7,90.,(),40.,radial_points=360,lmax=8)
    r = np.geomspace(.02,30.,60)
    old = IsothermalFluidHalo(3e7,90.)
    np.testing.assert_allclose(h.equatorial_circular_velocity(r),old.circular_velocity(r),rtol=.004)
    np.testing.assert_allclose(h.density(r,0.),h.density(0*r,r),rtol=1e-10)
    assert h.diagnostics['fixed_point_log_density_residual'] < h.tolerance
    assert h.density(0.,0.) == pytest.approx(h.rho0_msun_kpc3)


def test_spherical_baryon_limit_against_independent_coupled_ode():
    h = AxisymmetricFluidHalo(3e7,90.,(MiyamotoNagai(1e10,0.,2.),),50.,radial_points=400,lmax=8)
    old = CoupledSphericalFluidHalo(3e7,90.,PlummerBaryons(1e10,2.),50.)
    r = np.geomspace(.03,35.,60)
    np.testing.assert_allclose(h.equatorial_circular_velocity(r),old.circular_velocity(r),rtol=.004)
    np.testing.assert_allclose(h.density(r,0.),old.density(r),rtol=.006)


def test_axisymmetric_reflection_boltzmann_and_differential_poisson_closure():
    h = AxisymmetricFluidHalo(3e7,100.,(MiyamotoNagai(1e10,2.,.4),),50.,radial_points=440,lmax=16,angular_points=56)
    R = np.geomspace(.4,15.,25)
    z = np.full_like(R,.9)
    np.testing.assert_allclose(h.density(R,z),h.density(R,-z),rtol=1e-12)
    step = R*1e-4
    dp = (h.pressure(R+step,z)-h.pressure(R-step,z))/(2*step)
    dphi = (h.potential_offset(R+step,z)-h.potential_offset(R-step,z))/(2*step)
    np.testing.assert_allclose(dp,-h.density(R,z)*dphi,rtol=1e-5)
    np.testing.assert_allclose(h.poisson_density(R,z),h.density(R,z),rtol=.006)
    assert h.density(3.,0.) > h.density(0.,3.)
    assert h.diagnostics['minimum_boundary_pressure'] > 0


def test_nonconvergence_and_invalid_domains_fail_explicitly():
    with pytest.raises(RuntimeError,match='did not converge'):
        AxisymmetricFluidHalo(3e7,100.,(),20.,max_iterations=1,tolerance=1e-14).density(1.,0.)
    h = AxisymmetricFluidHalo(3e7,100.,(),20.)
    for R,z in [(-1.,0.),(21.,0.),(np.nan,0.)]:
        with pytest.raises(ValueError): h.density(R,z)
    with pytest.raises(ValueError): MiyamotoNagai(1.,1.,0.)
    with pytest.raises(ValueError): AxisymmetricFluidHalo(1.,1.,(),10.,lmax=3)


def test_vacuum_multipole_boundary_and_gauss_mass_closure():
    from numpy.polynomial.legendre import leggauss
    h = AxisymmetricFluidHalo(3e7,100.,(MiyamotoNagai(1e10,2.,.4),),50.,radial_points=500,lmax=16,angular_points=56)
    mu, weight = leggauss(32)
    r = h.outer_radius_kpc
    inner = h.halo_potential_offset(r*np.sqrt(1-mu**2),r*mu)+h._solution['central_phi']
    np.testing.assert_allclose(inner,h.exterior_halo_potential(r,mu),rtol=1e-12)
    step = .001*r
    derivative = (h.halo_potential_offset(r*np.sqrt(1-mu**2),r*mu)-h.halo_potential_offset((r-step)*np.sqrt(1-mu**2),(r-step)*mu))/step
    surface_mass = r*r/G*np.sum(weight*derivative)/2
    assert surface_mass == pytest.approx(h.diagnostics['total_halo_mass_inside_domain_msun'],rel=.003)
