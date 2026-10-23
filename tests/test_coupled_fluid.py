"""Independent physical closure and limiting cases for coupled baryon gravity."""
import numpy as np
import pytest
from scipy.integrate import quad

from galaxy_lab.coupled_fluid import (
    CoupledSphericalFluidHalo, HernquistBaryons, PlummerBaryons,
)
from galaxy_lab.fluid import G, IsothermalFluidHalo


@pytest.mark.parametrize("baryon_type", [PlummerBaryons, HernquistBaryons])
def test_zero_baryon_limit_matches_independently_tabulated_isothermal_halo(baryon_type):
    h = CoupledSphericalFluidHalo(3e7, 90., baryon_type(0., 2.), 150.)
    old = IsothermalFluidHalo(3e7, 90.)
    r = np.r_[0., np.geomspace(1e-8, 150., 170)]
    np.testing.assert_allclose(h.density(r), old.density(r), rtol=2e-8)
    # The reference's Hermite interpolation derivative sets an approximately
    # 4e-8 relative floor for mass, despite its tighter potential accuracy.
    np.testing.assert_allclose(h.enclosed_mass(r), old.enclosed_mass(r), rtol=1e-7, atol=1e-20)
    np.testing.assert_allclose(h.potential_offset(r), old.potential_offset(r), rtol=2e-8, atol=1e-15)
    np.testing.assert_allclose(h.circular_velocity(r), old.circular_velocity(r), rtol=5e-8)


@pytest.mark.parametrize("baryon_type", [PlummerBaryons, HernquistBaryons])
def test_hydrostatic_mass_and_total_potential_closure(baryon_type):
    h = CoupledSphericalFluidHalo(3e7, 90., baryon_type(4e10, 2.), 150.)
    r = np.geomspace(.003, 120., 110)
    dr = r*2e-4
    dp = (h.pressure(r+dr)-h.pressure(r-dr))/(2*dr)
    dm = (h.enclosed_mass(r+dr)-h.enclosed_mass(r-dr))/(2*dr)
    dphi = (h.potential_offset(r+dr)-h.potential_offset(r-dr))/(2*dr)
    grav = G*h.total_mass(r)/r**2
    np.testing.assert_allclose(dp, -h.density(r)*grav, rtol=8e-6)
    np.testing.assert_allclose(dm, 4*np.pi*r*r*h.density(r), rtol=8e-6)
    np.testing.assert_allclose(dphi, grav, rtol=8e-6)
    np.testing.assert_allclose(h.density(r)/h.rho0_msun_kpc3,
                               np.exp(-h.potential_offset(r)/h.sigma_kms**2), rtol=1e-13)
    assert h.density(0.) == h.rho0_msun_kpc3
    assert h.enclosed_mass(0.) == h.circular_velocity(0.) == 0.
    assert np.all(np.diff(h.density(r)) < 0)


@pytest.mark.parametrize("baryon_type", [PlummerBaryons, HernquistBaryons])
def test_baryon_potential_and_density_are_consistent_independently(baryon_type):
    b = baryon_type(4e10, 2.)
    r = np.geomspace(.001, 80., 50)
    step = r*1e-4
    dmass = (b.enclosed_mass(r+step)-b.enclosed_mass(r-step))/(2*step)
    dphi = (b.potential_offset(r+step)-b.potential_offset(r-step))/(2*step)
    np.testing.assert_allclose(dmass, 4*np.pi*r*r*b.density(r), rtol=2e-6)
    np.testing.assert_allclose(dphi, G*b.enclosed_mass(r)/r**2, rtol=2e-6)
    assert b.enclosed_mass(0.) == b.potential_offset(0.) == 0.


def test_coupled_mass_against_independent_density_quadrature():
    h = CoupledSphericalFluidHalo(8e7, 130., HernquistBaryons(3e10, 1.7), 100.)
    for r in (.03, .7, 5., 40.):
        m = quad(lambda x: 4*np.pi*x*x*h.density(x), 0., r,
                 epsabs=1e-5, epsrel=2e-8)[0]
        assert h.enclosed_mass(r) == pytest.approx(m, rel=2e-7)


def test_halo_responds_to_baryons_and_is_not_added_force_approximation():
    coupled = CoupledSphericalFluidHalo(3e7, 90., PlummerBaryons(4e10, 2.), 100.)
    alone = IsothermalFluidHalo(3e7, 90.)
    # Compare at fixed central density and sigma, not fixed total halo mass.
    assert coupled.density(2.) < .5*alone.density(2.)
    r = np.array([1., 2., 5.])
    uncontracted = np.sqrt(alone.circular_velocity(r)**2+G*coupled.baryons.enclosed_mass(r)/r)
    assert np.max(np.abs(uncontracted-coupled.circular_velocity(r))) > 5.


def test_coupled_model_retains_distance_scale_degeneracy():
    h = CoupledSphericalFluidHalo(3e7, 90., HernquistBaryons(4e10, 2.), 100.)
    scale = 3.
    other = CoupledSphericalFluidHalo(3e7/scale**2, 90.,
                                     HernquistBaryons(4e10*scale, 2.*scale), 100.*scale)
    r = np.geomspace(.01, 90., 90)
    np.testing.assert_allclose(h.circular_velocity(r), other.circular_velocity(scale*r), rtol=2e-9)


@pytest.mark.parametrize("kwargs", [dict(rho0_msun_kpc3=0.), dict(sigma_kms=-1.),
                                    dict(max_radius_kpc=np.inf), dict(rtol=0.)])
def test_invalid_halo_parameters(kwargs):
    base = dict(rho0_msun_kpc3=3e7, sigma_kms=90., baryons=PlummerBaryons(4e10, 2.))
    base.update(kwargs)
    with pytest.raises(ValueError):
        CoupledSphericalFluidHalo(**base)


def test_invalid_radius_and_baryon_distribution():
    h = CoupledSphericalFluidHalo(3e7, 90., PlummerBaryons(4e10, 2.), 100.)
    for r in (-1., np.nan, np.inf, 101.):
        with pytest.raises(ValueError):
            h.density(r)
    for b in (None, lambda r: r):
        with pytest.raises(TypeError):
            CoupledSphericalFluidHalo(3e7, 90., b)
    with pytest.raises(ValueError):
        PlummerBaryons(-2., 1.)
    with pytest.raises(ValueError):
        HernquistBaryons(1., 0.)
