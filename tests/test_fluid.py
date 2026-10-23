"""Physical closure tests, independent limiting cases, and unit checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from galaxy_lab.fluid import (
    G, GYR_S, KPC_M, MSUN_KG, IsothermalFluidHalo, cylindrical_shear,
    dimensionless_profile, fluid_circular_speed, kinematic_viscosity_m2_s,
    viscous_dissipation_w_m3, viscous_timescale_gyr,
)


def test_origin_and_uniform_core_limit():
    halo = IsothermalFluidHalo(2.0e7, 90.0)
    assert halo.density(0) == halo.rho0_msun_kpc3
    assert halo.enclosed_mass(0) == halo.circular_velocity(0) == 0
    radius = halo.scale_radius_kpc * np.geomspace(1e-7, 1e-3, 30)
    uniform_mass = 4 * np.pi * halo.rho0_msun_kpc3 * radius**3 / 3
    np.testing.assert_allclose(halo.enclosed_mass(radius), uniform_mass, rtol=2e-7)
    uniform_speed = np.sqrt(4 * np.pi * G * halo.rho0_msun_kpc3 / 3) * radius
    np.testing.assert_allclose(halo.circular_velocity(radius), uniform_speed, rtol=1e-7)


@pytest.mark.parametrize("x", [0.01, 0.2, 1.0, 8.0, 100.0, 1e4])
def test_enclosed_mass_against_independent_density_quadrature(x):
    halo = IsothermalFluidHalo(1.0e7, 70.0)
    radius = x * halo.scale_radius_kpc
    # Integrate dimensionless density, independently of the derivative/mass API.
    integral = quad(lambda y: np.exp(-dimensionless_profile(y)[0]) * y*y,
                    0, x, epsabs=1e-11, epsrel=1e-8, limit=300)[0]
    quadrature_mass = 4 * np.pi * halo.rho0_msun_kpc3 * halo.scale_radius_kpc**3 * integral
    np.testing.assert_allclose(halo.enclosed_mass(radius), quadrature_mass, rtol=2e-7)


def test_hydrostatic_and_poisson_closure_from_finite_differences():
    halo = IsothermalFluidHalo(3e7, 125)
    radius = halo.scale_radius_kpc * np.geomspace(0.003, 1e4, 60)
    step = radius * 2e-4
    pressure_gradient = (halo.pressure(radius + step) - halo.pressure(radius - step)) / (2 * step)
    gravity_force = -halo.density(radius) * G * halo.enclosed_mass(radius) / radius**2
    np.testing.assert_allclose(pressure_gradient, gravity_force, rtol=1e-5)
    mass_gradient = (halo.enclosed_mass(radius + step) - halo.enclosed_mass(radius - step)) / (2 * step)
    np.testing.assert_allclose(mass_gradient, 4*np.pi*radius**2*halo.density(radius), rtol=1e-5)


def test_asymptotic_speed_and_positive_monotonic_density():
    halo = IsothermalFluidHalo(1e8, 100)
    radius = halo.scale_radius_kpc * np.geomspace(1e-7, 1e8, 500)
    assert np.all(halo.density(radius) > 0)
    assert np.all(np.diff(halo.density(radius)) <= 0)
    assert np.all(np.diff(halo.enclosed_mass(radius)) > 0)
    np.testing.assert_allclose(halo.circular_velocity(radius[-1]), np.sqrt(2)*100, rtol=2e-4)


def test_distance_scale_degeneracy_and_optimizer_wrapper():
    # Exact scale freedom: R -> lambda*R, rho0 -> rho0/lambda², same sigma.
    halo = IsothermalFluidHalo(8e7, 110)
    radius = np.geomspace(.01, 100, 100)
    scale = 3.5
    rescaled = IsothermalFluidHalo(halo.rho0_msun_kpc3 / scale**2, halo.sigma_kms)
    np.testing.assert_allclose(halo.circular_velocity(radius), rescaled.circular_velocity(scale*radius), rtol=1e-13)
    np.testing.assert_allclose(fluid_circular_speed(radius, np.log10(8e7), np.log10(110)), halo.circular_velocity(radius), rtol=1e-13)


def test_correct_cylindrical_shear_and_nonnegative_dissipation():
    radius = np.array([1., 2., 10.])
    np.testing.assert_allclose(cylindrical_shear(radius, 30*radius, 30), 0)
    flat_shear = cylindrical_shear(radius, 200, 0)
    np.testing.assert_allclose(flat_shear, -200/radius)
    heating = viscous_dissipation_w_m3(flat_shear, .1)
    np.testing.assert_allclose(heating, .1*(flat_shear*1000/KPC_M)**2)
    assert np.all(heating > 0)
    np.testing.assert_array_equal(viscous_dissipation_w_m3(flat_shear, 0), 0)


def test_viscous_units_and_timescale():
    rho = 1e7
    mu = 1e-5
    expected_nu = mu / (rho * MSUN_KG / KPC_M**3)
    assert kinematic_viscosity_m2_s(mu, rho) == pytest.approx(expected_nu)
    assert viscous_timescale_gyr(5, expected_nu) == pytest.approx((5*KPC_M)**2/expected_nu/GYR_S)
    assert np.isinf(viscous_timescale_gyr(5, 0))


@pytest.mark.parametrize("rho,sigma", [(0,100), (-1,100), (1e7,0), (np.inf,10), (1,np.nan)])
def test_invalid_physical_parameters(rho,sigma):
    with pytest.raises(ValueError):
        IsothermalFluidHalo(rho,sigma)


def test_invalid_domains_are_rejected():
    halo = IsothermalFluidHalo(1e7,100)
    for radius in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            halo.circular_velocity(radius)
    with pytest.raises(ValueError):
        dimensionless_profile(1.01e8)
    with pytest.raises(ValueError):
        cylindrical_shear(0, 0, 1)
    with pytest.raises(ValueError):
        viscous_dissipation_w_m3(1, -1)
    with pytest.raises(ValueError):
        viscous_timescale_gyr(0, 0)
