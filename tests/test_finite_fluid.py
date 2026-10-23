import numpy as np
import pytest
from scipy.integrate import quad
from galaxy_lab.finite_fluid import (king_moment,SphericalFiniteHalo,FiniteFluidHalo,
                                     MiyamotoNagai,G)


def test_king_density_and_pressure_are_independent_velocity_integrals():
    for W in (1e-5,.1,1.,5.,9.):
        upper=np.sqrt(2*W)
        density=4*np.pi/(2*np.pi)**1.5*quad(lambda u:np.expm1(W-u*u/2)*u*u,0,upper,epsabs=1e-25)[0]
        pressure=4*np.pi/(3*(2*np.pi)**1.5)*quad(lambda u:np.expm1(W-u*u/2)*u**4,0,upper,epsabs=1e-25)[0]
        assert np.isclose(king_moment(W),density,rtol=2e-10)
        assert np.isclose(king_moment(W,True),pressure,rtol=2e-10)
    assert np.array_equal(king_moment([-1.,0.]),[0.,0.])


def test_pressure_derivative_supplies_density_and_nonisothermal_dispersion():
    W=np.geomspace(.001,9,50);step=W*1e-5
    derivative=(king_moment(W+step,True)-king_moment(W-step,True))/(2*step)
    assert np.allclose(derivative,king_moment(W),rtol=1e-7)
    ratio=king_moment(W,True)/king_moment(W)
    assert np.all((ratio>0)&(ratio<1))
    assert np.all(np.diff(ratio)>0)


def test_spherical_reference_has_finite_mass_free_edge_and_vacuum_force():
    h=SphericalFiniteHalo(3e7,100.,5.)
    edge=h.edge_radius_kpc
    assert 70<edge<90
    assert h.density(edge*1.01)==0
    assert h.pressure(edge*1.01)==0
    r=np.array([1.2,2.,5.])*edge
    assert np.allclose(h.circular_velocity(r)**2,G*h.total_mass_msun/r,rtol=1e-12)


def test_axisymmetric_solver_matches_independent_spherical_ode():
    h=FiniteFluidHalo(3e7,100.,(),radial_points=640,lmax=8,angular_points=24,tolerance=1e-9)
    ref=SphericalFiniteHalo(3e7,100.,5.)
    r=np.geomspace(.2,ref.edge_radius_kpc*1.5,100)
    assert np.max(abs(h.equatorial_circular_velocity(r)/ref.circular_velocity(r)-1))<4e-4
    assert abs(h.diagnostics['total_halo_mass_msun']/ref.total_mass_msun-1)<4e-4


def test_flattened_poisson_and_hydrostatic_closure_and_surface():
    h=FiniteFluidHalo(3e7,100.,(MiyamotoNagai(1e10,2.,.4),),radial_points=640,lmax=24,angular_points=72,tolerance=1e-9)
    r=np.geomspace(.3,20.,25);mu=np.linspace(0,.9,7)
    R=r[:,None]*np.sqrt(1-mu**2);z=r[:,None]*mu
    assert np.max(abs(h.poisson_density(R,z)/h.density(R,z)-1))<.002
    step=1e-4
    pressure_grad=(h.pressure(R+step,z)-h.pressure(R-step,z))/(2*step)
    potential_grad=(h.potential_offset(R+step,z)-h.potential_offset(R-step,z))/(2*step)
    assert np.max(abs(pressure_grad/(h.density(R,z)*potential_grad)+1))<2e-6
    assert h.diagnostics['maximum_boundary_pressure']==0
    assert h.diagnostics['cutoff_energy_kms2']<0
    outer=h.diagnostics['outer_radius_kpc']
    angles=np.linspace(0,1,20)
    assert np.all(h.density(outer*np.sqrt(1-angles**2),outer*angles)==0)
    assert np.allclose(h.density(R,z),h.density(R,-z))


def test_vacuum_potential_matches_and_wrong_domain_is_rejected():
    h=FiniteFluidHalo(3e7,100.,(),100.,320,8,24,tolerance=1e-9)
    outer=h.diagnostics['outer_radius_kpc'];mu=np.linspace(-1,1,15)
    interior=h.halo_potential_offset(outer*np.sqrt(1-mu**2),outer*mu)+h._solution['central_phi']
    assert np.allclose(interior,h.exterior_halo_potential(outer,mu),rtol=1e-12,atol=1e-8)
    far=outer*100
    assert np.isclose(h.equatorial_circular_velocity(far)**2,G*h.diagnostics['total_halo_mass_msun']/far,rtol=1e-10)
    assert h.density(far,0)==0 and h.pressure(far,0)==0
    with pytest.raises(RuntimeError,match='does not enclose'):
        FiniteFluidHalo(3e7,100.,(),20.,160,8,24).diagnostics


def test_distance_similarity_does_not_identify_absolute_scale():
    b=(MiyamotoNagai(1e10,2.,.4),);factor=3.
    h=FiniteFluidHalo(3e7,100.,b,radial_points=240,lmax=12,angular_points=36)
    hs=FiniteFluidHalo(3e7/factor**2,100.,(MiyamotoNagai(1e10*factor,2*factor,.4*factor),),radial_points=240,lmax=12,angular_points=36)
    r=np.geomspace(.3,20.,30)
    assert np.allclose(h.equatorial_circular_velocity(r),hs.equatorial_circular_velocity(r*factor),rtol=1e-10)


def test_invalid_parameters_and_nonconvergence_raise():
    with pytest.raises(ValueError):SphericalFiniteHalo(-1,100.)
    with pytest.raises(ValueError):SphericalFiniteHalo(1e7,100.,0)
    with pytest.raises(ValueError):king_moment(np.nan)
    with pytest.raises(RuntimeError,match='did not converge'):
        FiniteFluidHalo(3e7,100.,(),radial_points=100,lmax=4,angular_points=12,max_iterations=1).diagnostics
