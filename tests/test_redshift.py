import numpy as np
import pandas as pd
import pytest
from galaxy_lab.redshift import (observable_frame, assert_independent_training_descriptors,
                                cosmological_distances)


def test_observed_photometry_does_not_use_catalog_redshift_or_absolute_values():
    source = pd.DataFrame({**{"flux_"+b: [100., 10.] for b in "ugriz"},
        "half_light_radius_arcsec": [5., 8.], "axis_ratio": [.7, .9],
        "spec_z": [.01, .1], "absolute_mag": [-20., -22.]})
    a = observable_frame(source)
    source["spec_z"] = [1000., 1e8]; source["absolute_mag"] = -2000
    b = observable_frame(source)
    pd.testing.assert_frame_equal(a, b)
    assert a.mag_r.iloc[0] == pytest.approx(17.5)


def test_in_sample_arm_features_rejected_for_distance_training():
    rows = pd.DataFrame({"id": ["a", "b", "c"], "sky_group": ["1", "2", "3"],
        "split": ["train", "calibration", "test"],
        "prediction_source": ["in_sample", "train_only_model_calibration", "frozen_model_test"]})
    with pytest.raises(ValueError, match="out-of-fold"):
        assert_independent_training_descriptors(rows)
    rows.loc[0, "prediction_source"] = "out_of_fold_train"
    assert_independent_training_descriptors(rows)
    rows.loc[1, "sky_group"] = "1"
    with pytest.raises(ValueError, match="cannot cross"):
        assert_independent_training_descriptors(rows)


def test_cosmological_distance_is_explicitly_conditional_and_not_z_times_c_over_h0():
    result = cosmological_distances([.05], [.02], [.08])
    assert result["status"].startswith("conditional")
    assert result["luminosity_distance_mpc"][0] > result["angular_diameter_distance_mpc"][0]
    assert result["conditional_angular_diameter_lower_mpc"][0] < result["angular_diameter_distance_mpc"][0] < result["conditional_angular_diameter_upper_mpc"][0]
    with pytest.raises(ValueError):
        cosmological_distances([1.])
