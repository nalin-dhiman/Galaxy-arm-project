import numpy as np
import pytest
from galaxy_lab.spectral_kinematics import DiskGeometry,annular_los_fit,assess_flat_speed


def fixture_map():
    y,x=np.mgrid[-24:24:.5,-24:24:.5]
    g=DiskGeometry(.65,37.)
    r,c,_=g.disk_coordinates(x,y)
    v=23.+np.sin(np.radians(g.inclination_deg))*c*130.
    good=(r<25)&(r>0)
    return x,y,v,np.ones_like(v)/4,good,g


def test_recovers_angular_los_rotation_without_distance():
    x,y,v,w,good,g=fixture_map()
    a=annular_los_fit(x,y,v,w,good,g,psf_fwhm_arcsec=2.5)
    np.testing.assert_allclose(a['systemic_offset_kms'],23,atol=1e-10)
    np.testing.assert_allclose([r['rotation_kms'] for r in a['rings']],130,atol=1e-10)
    assert a['residual_rms_kms']<1e-10


def test_systemic_shift_does_not_change_rotation():
    x,y,v,w,good,g=fixture_map()
    a=annular_los_fit(x,y,v+400,w,good,g,psf_fwhm_arcsec=2.5)
    np.testing.assert_allclose([r['rotation_kms'] for r in a['rings']],130,atol=1e-9)


def test_sky_rotation_and_pa_covariance():
    x,y,v,w,good,g=fixture_map();a=np.radians(29)
    # Rotate north toward east, increasing astronomical PA by29degrees.
    b=annular_los_fit(x*np.cos(a)+y*np.sin(a),y*np.cos(a)-x*np.sin(a),v,w,good,DiskGeometry(.65,66.),psf_fwhm_arcsec=2.5)
    np.testing.assert_allclose([r['rotation_kms'] for r in b['rings']],130,atol=1e-9)


def test_bad_geometry_and_unresolved_bins_rejected():
    with pytest.raises(ValueError):DiskGeometry(.1,0).inclination_deg
    x,y,v,w,good,g=fixture_map()
    with pytest.raises(ValueError):annular_los_fit(x,y,v,w,good,g,psf_fwhm_arcsec=4)
    with pytest.raises(ValueError):annular_los_fit(x,y,v,w,np.zeros_like(good),g,psf_fwhm_arcsec=2)


def test_masked_side_and_quality_force_abstention():
    x,y,v,w,good,g=fixture_map()
    a=annular_los_fit(x,y,v,w,good,g,psf_fwhm_arcsec=2)
    assert assess_flat_speed(a,half_light_radius_arcsec=5)['optical_flatness_candidate']
    b=assess_flat_speed(a,half_light_radius_arcsec=50,quality_flags=1)
    assert not b['optical_flatness_candidate']
    assert 'radial_extent_below_three_inferred_disk_scale_lengths' in b['reasons']
    assert 'nonzero_observation_quality_flags' in b['reasons']
