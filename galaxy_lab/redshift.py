"""Calibrated statistical redshift inference from observed galaxy features.

These models estimate inherited spectroscopic catalog redshift in a selected
nearby-spiral population. They are not geometric standard rulers. Optional
cosmological distances are conditional conversions, with no distance coverage
guarantee or deconvolution of peculiar velocities.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .distance import validate_feature_names

PHOTO = ["mag_" + b for b in "ugriz"] + ["color_" + a + "_" + b for a, b in zip("ugri", "griz")]
GEOMETRY = ["angular_half_light_radius_arcsec", "axis_ratio"]
IMAGE_SOURCE = ["g_annular_positive_flux", "r_annular_positive_flux", "z_annular_positive_flux",
    "r_annular_r50_arcsec", "r_relative_contrast_rms", "gr_annular_asinh_color", "rz_annular_asinh_color"]
IMAGE = ["apparent_img_" + key for key in IMAGE_SOURCE]
ARM = ["arm_probability_mean", "arm_probability_p90", "arm_probability_std",
    "arm_probability_weighted_radius_fraction", "arm_probability_inner_mean",
    "arm_probability_outer_mean", "arm_probability_entropy_mean", "arm_candidate_area_fraction"]
NUISANCE = (["apparent_mag_error_" + b for b in "ugriz"]
    + ["psf_" + b + "_fwhm_arcsec" for b in "grz"]
    + ["apparent_noise_" + b + "_nanomaggy" for b in "grz"])
FAMILIES = {"P": PHOTO + NUISANCE, "PG": PHOTO + NUISANCE + GEOMETRY,
            "PGI": PHOTO + NUISANCE + GEOMETRY + IMAGE,
            "PGIA": PHOTO + NUISANCE + GEOMETRY + IMAGE + ARM}


def observable_frame(catalog, descriptors=None, quality=None):
    """Convert observed nanomaggy photometry; preserve no label-derived features."""
    result = pd.DataFrame(index=catalog.index)
    for band in "ugriz":
        flux = pd.to_numeric(catalog["flux_" + band], errors="raise").to_numpy(float)
        good = np.isfinite(flux) & (flux > 0)
        result["mag_" + band] = np.where(good, 22.5 - 2.5 * np.log10(np.where(good, flux, 1)), np.nan)
        ivar = pd.to_numeric(catalog.get("flux_ivar_" + band,
            pd.Series(np.nan, index=catalog.index)), errors="raise").to_numpy(float)
        noise_good = good & np.isfinite(ivar) & (ivar > 0)
        denom = np.where(noise_good, flux * np.sqrt(np.maximum(ivar, 0)), 1.)
        result["apparent_mag_error_" + band] = np.where(noise_good, 2.5 / np.log(10) / denom, np.nan)
    for a, b in zip("ugri", "griz"):
        result["color_" + a + "_" + b] = result["mag_" + a] - result["mag_" + b]
    result[GEOMETRY[0]] = pd.to_numeric(catalog["half_light_radius_arcsec"], errors="raise")
    result[GEOMETRY[1]] = pd.to_numeric(catalog["axis_ratio"], errors="raise")
    for key in NUISANCE[5:]:
        result[key] = np.nan if quality is None else pd.to_numeric(quality[key], errors="raise").to_numpy(float)
    if descriptors is not None:
        if len(descriptors) != len(result):
            raise ValueError("Catalog and descriptor rows must be explicitly aligned")
        for source, key in zip(IMAGE_SOURCE, IMAGE):
            result[key] = pd.to_numeric(descriptors[source], errors="raise").to_numpy(float)
        for key in ARM:
            result[key] = pd.to_numeric(descriptors[key], errors="raise").to_numpy(float)
    return result


def assert_independent_training_descriptors(frame):
    """Training arm measurements must have been predicted without their group."""
    if frame["id"].duplicated().any():
        raise ValueError("One descriptor row per galaxy required")
    if frame.groupby("sky_group").split.nunique().max() != 1:
        raise ValueError("Sky groups cannot cross train/calibration/test boundaries")
    train = frame.split == "train"
    if not train.any() or not (frame.loc[train, "prediction_source"] == "out_of_fold_train").all():
        raise ValueError("Every training arm descriptor must be out-of-fold by sky group")
    expected = {"calibration": "train_only_model_calibration", "test": "frozen_model_test"}
    for split, source in expected.items():
        if not (frame.loc[frame.split == split, "prediction_source"] == source).all():
            raise ValueError(split + " descriptors must come from the frozen training model")


@dataclass
class CalibratedRedshiftModel:
    estimator: object
    features: list[str]
    conformal_radius_log1pz: float
    training_min: dict
    training_max: dict
    metadata: dict

    def __post_init__(self):
        extras = [x for x in self.features if x not in PHOTO]
        # Name guards supplement an explicit provenance schema; they do not
        # prove that an apparently observable number was independently measured.
        validate_feature_names(PHOTO, extras or ["angular_unused_schema_guard"])
        if not self.features or len(self.features) != len(set(self.features)):
            raise ValueError("A distinct nonempty predictor list is required")
        if self.conformal_radius_log1pz < 0 or np.isnan(self.conformal_radius_log1pz):
            raise ValueError("Conformal radius must be nonnegative")

    def predict(self, features):
        missing = [key for key in self.features if key not in features]
        if missing:
            raise ValueError("Missing observed predictors: " + ", ".join(missing))
        x = features[self.features].astype(float)
        if np.isinf(x.to_numpy()).any():
            raise ValueError("Infinite predictors are invalid")
        center = self.estimator.predict(x)
        radius = self.conformal_radius_log1pz
        with np.errstate(over="ignore"):
            prediction = np.maximum(0., np.expm1(center))
            lower = np.maximum(0., np.expm1(center - radius))
            upper = np.expm1(center + radius)
        outside = np.zeros(len(x), bool)
        missing_count = np.zeros(len(x), int)
        for key in self.features:
            values = x[key].to_numpy()
            outside |= (values < self.training_min[key]) | (values > self.training_max[key])
            missing_count += ~np.isfinite(values)
        return pd.DataFrame({"predicted_catalog_redshift": prediction,
            "redshift_interval_lower": lower, "redshift_interval_upper": upper,
            "outside_training_feature_range": outside, "missing_predictors": missing_count,
            "status": "experimental_selected_population_model"}, index=features.index)

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "redshift_model.joblib", compress=3)
        schema = {"features": self.features, "conformal_radius_log1pz": self.conformal_radius_log1pz,
            "training_min": self.training_min, "training_max": self.training_max, **self.metadata,
            "input_scope": "Observed calibrated photometry and declared image-only descriptors. No target, physical size, mass or fluid-fit quantities.",
            "interval_scope": "Catalog-redshift residual calibration for comparable exchangeable sky groups. Not calibrated geometric-distance uncertainty.",
            "artifact_loading": "Load only trusted locally generated joblib artifacts"}
        (directory / "model_card.json").write_text(json.dumps(schema, indent=2, allow_nan=False) + "\n")


def cosmological_distances(predicted_z, lower_z=None, upper_z=None):
    """Planck18 conversion for local z<=.3, explicitly conditional on Hubble flow.

    The redshift interval is transformed monotonically in this limited domain.
    It does not cover peculiar-velocity, cosmology or redshift-label systematics.
    """
    from astropy.cosmology import Planck18
    values = np.atleast_1d(np.asarray(predicted_z, float))
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > .3):
        raise ValueError("Conditional conversion restricted to finite 0<=z<=0.3")
    result = {"cosmology": "Planck18", "angular_diameter_distance_mpc": Planck18.angular_diameter_distance(values).value.tolist(),
        "luminosity_distance_mpc": Planck18.luminosity_distance(values).value.tolist(),
        "status": "conditional_cosmological_conversion_not_geometric_distance_validation",
        "limitations": "Assumes cosmological Hubble-flow redshift; peculiar velocities, observer-frame correction and cosmology uncertainty are not inferred."}
    if (lower_z is None) != (upper_z is None):
        raise ValueError("Both redshift interval endpoints are required")
    if lower_z is not None:
        lo, hi = np.broadcast_arrays(np.asarray(lower_z, float), np.asarray(upper_z, float))
        if lo.shape != values.shape or not np.isfinite([lo, hi]).all() or np.any(lo < 0) or np.any(hi > .3) or np.any(lo > hi):
            raise ValueError("Finite ordered matching intervals within0<=z<=0.3 required")
        result["conditional_angular_diameter_lower_mpc"] = Planck18.angular_diameter_distance(lo).value.tolist()
        result["conditional_angular_diameter_upper_mpc"] = Planck18.angular_diameter_distance(hi).value.tolist()
    return result
