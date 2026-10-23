import numpy as np
import pytest

from galaxy_lab.distance import (
    conformal_radius, evaluate_distance_features, group_disjoint_split,
    synthetic_distance_demo, validate_feature_names,
)


def test_group_split_no_leaks_deterministic_and_row_order_invariant():
    groups = np.repeat([f"group_{i}" for i in range(30)],3)
    split = group_disjoint_split(groups)
    assert np.array_equal(split,group_disjoint_split(groups))
    for group in np.unique(groups):
        assert len(set(split[groups==group]))==1
    assert set(split)=={"train","calibration","test"}
    perm = np.random.default_rng(1).permutation(len(groups))
    np.testing.assert_array_equal(group_disjoint_split(groups[perm]),split[perm])


def test_finite_sample_conformal_order_statistic_and_infinity():
    assert conformal_radius(np.arange(1,10),alpha=.1)==9
    assert conformal_radius([1,2,3],alpha=.5)==2
    assert np.isinf(conformal_radius([1,2,3],alpha=.1))
    with pytest.raises(ValueError):
        conformal_radius([],alpha=.1)


@pytest.mark.parametrize("bad",["distance_mpc","angular_radius_kpc","apparent_absolute_mag",
                                "arm_mass","flux_luminosity","pitch_viscosity","color_spec_z",
                                "angular_velocity","galaxy_id","surface_brightness_lsun_pc2",
                                "angular_area_kpc2","foo"])
def test_reject_target_and_physical_feature_leakage(bad):
    with pytest.raises(ValueError):
        validate_feature_names(["apparent_mag_r"],[bad])


def test_z_band_is_an_allowed_photometric_observable():
    photo,morph = validate_feature_names(["mag_z","color_i_z"],["angular_radius_arcsec"])
    assert photo==["mag_z","color_i_z"]


def test_engineered_synthetic_recovery_uses_same_test_galaxies():
    data = synthetic_distance_demo(n_groups=200)
    summary,predictions,manifest = evaluate_distance_features(
        data,["apparent_mag_r","color_g_r"],["pitch_deg","arm_contrast"],
        synthetic=True,n_estimators=64,n_bootstrap=300)
    assert summary["experiment_kind"]=="SYNTHETIC DEMONSTRATION ONLY"
    errors = summary["models"]
    assert errors["photometry_morphology"]["mae_transformed_target"] < .6*errors["photometry"]["mae_transformed_target"]
    assert summary["paired_morphology_improvement"]["ci95_low"]>0
    assert set(predictions.galaxy_id)==set(manifest.loc[manifest.split=="test","galaxy_id"])
    assert manifest.groupby("group_id").split.nunique().max()==1


def test_imputation_and_model_training_do_not_use_test_labels_or_features():
    data = synthetic_distance_demo(n_groups=40)
    split = group_disjoint_split(data.group_id)
    feature = "apparent_mag_r"
    # Leave a training missing value and make heldout values extreme.
    data.loc[np.flatnonzero(split=="train")[0],feature] = np.nan
    original,_,_ = evaluate_distance_features(data,[feature],["pitch_deg"],n_estimators=16,n_bootstrap=100,
                                              missing_indicators=False)
    train_median = data.loc[split=="train",feature].median()
    assert original["models"]["photometry"]["train_only_imputation_medians"][feature]==train_median
    changed = data.copy()
    changed.loc[split=="test",feature]=1e20
    changed.loc[split=="test","distance_mpc"]*=10
    result,_,_ = evaluate_distance_features(changed,[feature],["pitch_deg"],n_estimators=16,n_bootstrap=100,
                                            missing_indicators=False)
    assert result["models"]["photometry"]["train_only_imputation_medians"][feature]==train_median
    assert original["models"]["photometry"]["conformal_radius_transformed"]==result["models"]["photometry"]["conformal_radius_transformed"]


def test_calibration_uses_group_maxima_and_small_group_count_gives_infinity():
    data = synthetic_distance_demo(n_groups=60,galaxies_per_group=3)
    summary,_,manifest = evaluate_distance_features(data,["apparent_mag_r"],["pitch_deg"],n_estimators=16,n_bootstrap=100)
    train = manifest.split=="train"
    cal = manifest.split=="calibration"
    median = np.median(np.log(data.loc[train,"distance_mpc"]))
    scores = np.abs(np.log(data.loc[cal,"distance_mpc"])-median)
    group_scores = scores.groupby(data.loc[cal,"group_id"]).max()
    assert summary["models"]["median"]["conformal_radius_transformed"]==conformal_radius(group_scores.to_numpy())
    tiny = synthetic_distance_demo(n_groups=5)
    tiny_summary,predictions,_ = evaluate_distance_features(tiny,["apparent_mag_r"],["pitch_deg"],n_estimators=8,n_bootstrap=100)
    assert np.isinf(tiny_summary["models"]["median"]["conformal_radius_transformed"])
    assert np.isinf(predictions.median_upper).all()
    assert (predictions.median_lower==0).all()
    assert tiny_summary["paired_morphology_improvement"]["ci95_low"] is None
    assert tiny_summary["paired_morphology_improvement"]["ci95_high"] is None
    assert tiny_summary["paired_morphology_improvement"]["bootstrap_draws"]==0


def test_redshift_target_not_implicitly_distance_and_duplicate_galaxies_rejected():
    data = synthetic_distance_demo(n_groups=40)
    data["spec_z"] = data.distance_mpc/4000
    summary,_,_ = evaluate_distance_features(data,["apparent_mag_r"],["pitch_deg"],target="spec_z",n_estimators=16,n_bootstrap=100)
    assert summary["distance_from_redshift_conversion"]=="none"
    assert summary["target"]=="spec_z"
    data.loc[1,"galaxy_id"]=data.loc[0,"galaxy_id"]
    with pytest.raises(ValueError):
        evaluate_distance_features(data,["apparent_mag_r"],["pitch_deg"],n_estimators=16,n_bootstrap=100)
