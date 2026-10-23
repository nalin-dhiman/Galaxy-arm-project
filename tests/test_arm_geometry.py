import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from galaxy_lab.filtering import ScienceImage,DiskGeometry
from galaxy_lab.arm_geometry import (deproject_points,project_points,geometry_from_builder,
    fit_log_spiral,trace_bootstrap_pitch,centerline_metrics,resample_curve,
    measure_cross_section_width,extract_centerlines,ExtractionConfig)


def geom(inc=0):return DiskGeometry((80.,75.),31.,inc,2.,20.)
def spiral(g,pitch=23.):
    t=np.linspace(-.5,2.,200);r=15*np.exp(np.tan(np.radians(pitch))*t)
    return project_points(np.c_[r*np.cos(t),r*np.sin(t)],g)


def test_projection_pitch_and_major_axis_convention():
    for inc in [0,30,60,72]:
        g=geom(inc);p=spiral(g)
        assert np.max(np.abs(project_points(deproject_points(p,g),g)-p))<1e-10
        assert fit_log_spiral(p,g)['pitch_deg']==pytest.approx(23,abs=1e-7)
    phi=.7;q=.6;g=geometry_from_builder((0,0),phi,q,1,20)
    p=np.array([[8.,10.],[12.,4.]])
    author=(np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])@p.T).T/np.array([q,1])
    assert np.allclose(np.linalg.norm(deproject_points(p,g),axis=1),np.linalg.norm(author,axis=1))


def test_short_trace_and_invalid_coordinates_rejected():
    with pytest.raises(ValueError):fit_log_spiral(np.c_[np.linspace(10,30,20)+80,np.full(20,75)],geom())
    with pytest.raises(ValueError):deproject_points([[np.nan,1]],geom())
    with pytest.raises(ValueError):resample_curve([[1,1],[1,1]])


def test_trace_bootstrap_resamples_drawings():
    traces=[spiral(geom(),p) for p in [20,21,22,23,24]]
    result=trace_bootstrap_pitch(traces,geom(),seed=3)
    assert result['trace_count']==5
    assert result['median_pitch_deg']==pytest.approx(22)
    assert result['interval95_deg'][0]<22<result['interval95_deg'][1]


def test_length_uniform_metrics_and_false_segments_penalty():
    p=spiral(geom());r=centerline_metrics([p],[p[::4]],.4,.4)
    assert r['f1']>.98
    p2=p+[35,35];r2=centerline_metrics([p,p2],[p],.4,.4)
    assert r2['precision']<.6 and r2['recall']>.99
    assert centerline_metrics([], [p],.4,.4)['f1']==0


def test_gaussian_width_psf_deconvolution_and_units():
    yy,xx=np.indices((180,180));scale=.2;intrinsic=1.;psf=.8
    sigma=np.hypot(intrinsic,psf)/2.354820045
    data=np.exp(-.5*((yy-90)*scale/sigma)**2)+.02*xx
    image=ScienceImage(data,scale,psf_fwhm_arcsec=psf)
    curve=np.c_[np.linspace(30,150,100),np.full(100,90.)]
    result=measure_cross_section_width(image,curve,half_width_arcsec=3)
    assert result['status']=='fit'
    assert result['deconvolved_fwhm_arcsec']==pytest.approx(intrinsic,abs=.03)


def test_smooth_and_noise_controls_not_automatically_spiral():
    rng=np.random.default_rng(4);yy,xx=np.indices((128,128));g=DiskGeometry((64,64),0,0,4,20)
    r=np.hypot(xx-64,yy-64);data=np.exp(-r/20)+rng.normal(0,.001,(128,128))
    curves,meta=extract_centerlines(ScienceImage(data,.4),g,ExtractionConfig(minimum_score=1.8))
    assert len(curves)==0
