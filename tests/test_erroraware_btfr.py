import numpy as np
import pytest
from galaxy_lab.erroraware_btfr import ErrorAwareBTFR


def model():
    v=np.geomspace(30,300,40);y=10+4*np.log10(v/100)
    return ErrorAwareBTFR.fit(v,y,np.full(40,.01),np.full(40,.05),grid_size=(101,81))


def test_known_relation_and_positive_conditional_distance():
    m=model();assert abs(m.summarize()['slope_5_50_95'][1]-4)<.12
    p=m.predict(100,1e8,.01,.02,draws=4000)
    assert abs(p['distance_mpc']-10)<.4
    assert 0<p['model_interval_90_lower_mpc']<p['distance_mpc']<p['model_interval_90_upper_mpc']


def test_flux_scaling_and_reference_independence():
    m=model();a=m.distance_draws(100,1e8,.01,.02);b=m.distance_draws(100,4e8,.01,.02)
    np.testing.assert_allclose(a/2,b,rtol=1e-14)


def test_more_measurement_uncertainty_broadens_interval():
    m=model();a=m.predict(100,1e8,.001,.001,draws=12000);b=m.predict(100,1e8,.1,.2,draws=12000)
    assert b['model_interval_90_upper_mpc']/b['model_interval_90_lower_mpc']>2*(a['model_interval_90_upper_mpc']/a['model_interval_90_lower_mpc'])


def test_velocity_definition_and_extrapolation_are_explicit():
    m=model()
    with pytest.raises(ValueError):m.predict(100,1e8,.01,.02,velocity_definition='HI_Wm50_over2_kms')
    assert m.predict(500,1e8,.01,.02)['status']=='abstain_velocity_extrapolation'


def test_numerical_artifact_roundtrip(tmp_path):
    m=model();p=tmp_path/'model.npz';m.save(p);n=ErrorAwareBTFR.load(p)
    assert m.predict(100,1e8,.01,.02)==n.predict(100,1e8,.01,.02)
    with pytest.raises(FileExistsError):m.save(p)


def test_invalid_measurements_rejected():
    m=model()
    with pytest.raises(ValueError):m.predict(-1,1e8,.01,.02)
    with pytest.raises(ValueError):m.predict(100,1e8,-.1,.02)


def test_unsuffixed_save_does_not_overwrite_npz(tmp_path):
    m=model();p=tmp_path/'model';m.save(p)
    assert p.with_suffix('.npz').exists()
    with pytest.raises(FileExistsError):m.save(p)
