#!/usr/bin/env python3
"""Frozen-split redshift ablation using out-of-fold learned arm descriptors."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/solid_galaxy_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.distance import conformal_radius, paired_group_bootstrap
from galaxy_lab.redshift import (PHOTO, NUISANCE, FAMILIES, observable_frame,
    assert_independent_training_descriptors, CalibratedRedshiftModel,
    cosmological_distances)
OUT = ROOT / "executed_solid/redshift"
DATA = ROOT / "executed_solid/data"
SEED = 20261023


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimator():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                            max_features=1., random_state=SEED, n_jobs=1))


def native_quality(catalog):
    manifest = json.loads((DATA / "native_images/manifest.json").read_text())
    lookup = {r["id"]: r for r in manifest.get("images", [])}
    rows = []
    for identifier in catalog.id:
        record = lookup.get(identifier, {})
        bands = record.get("bands", {})
        if not isinstance(bands, dict):
            bands = {r.get("name", r.get("band")): r for r in bands}
        row = {}
        for band in "grz":
            plane = bands.get(band, {})
            psf = record.get("psf", {}).get("bands", {}).get(band, {})
            row["psf_" + band + "_fwhm_arcsec"] = plane.get("psf_fwhm_arcsec", psf.get("gaussian_equivalent_fwhm_arcsec", np.nan))
            row["apparent_noise_" + band + "_nanomaggy"] = plane.get("median_sigma_nanomaggy", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare():
    catalog = pd.read_csv(DATA / "matched_catalog.csv")
    descriptors = pd.read_csv(ROOT / "executed_solid/segmentation/image_descriptors.csv")
    assert_independent_training_descriptors(descriptors)
    if catalog.id.duplicated().any():
        raise ValueError("Duplicate catalog galaxy")
    # Join by identity, never by incidental filesystem or download order.
    descriptors = descriptors.set_index("id").reindex(catalog.id).reset_index()
    for column in ("split", "sky_group"):
        present = descriptors[column].notna()
        if not (descriptors.loc[present, column].to_numpy() == catalog.loc[present, column].to_numpy()).all():
            raise ValueError("Descriptor and catalog split/sky-group mismatch")
    y = catalog.spec_z.to_numpy(float)
    label_good = np.isfinite(y) & (y >= 0) & (y <= .3)
    exclusions = catalog.loc[~label_good, ["id", "split", "sky_group", "spec_z"]].copy()
    exclusions["reason"] = "missing or outside predeclared local redshift domain0<=z<=0.3"
    quality = native_quality(catalog)
    features = observable_frame(catalog, descriptors, quality)
    features["apparent_arm_features_available"] = descriptors.prediction_source.notna().astype(float)
    for name in FAMILIES:
        if "apparent_arm_features_available" not in FAMILIES[name]:
            FAMILIES[name] = FAMILIES[name] + ["apparent_arm_features_available"]
    combined = pd.concat([catalog[["id", "split", "sky_group", "spec_z", "drp3qual"]], features], axis=1)
    combined = combined.loc[label_good].reset_index(drop=True)
    return combined, exclusions


def evaluate_predictions(frame, prediction, lo, hi):
    y = frame.spec_z.to_numpy(float)
    error = prediction - y
    normalized = error / (1 + y)
    covered = (y >= lo) & (y <= hi)
    grouped = pd.DataFrame({"group": frame.sky_group.to_numpy(), "covered": covered})
    return {"galaxies": len(y), "sky_groups": frame.sky_group.nunique(),
        "mae_z": float(np.mean(abs(error))), "median_absolute_error_z": float(np.median(abs(error))),
        "mean_signed_error_z": float(np.mean(error)),
        "sigma_nmad_delta_z_over_1plusz": float(1.4826 * np.median(abs(normalized - np.median(normalized)))),
        "outlier_fraction_abs_dz_over1pz_gt0p15": float(np.mean(abs(normalized) > .15)),
        "galaxy_interval_coverage": float(np.mean(covered)),
        "whole_group_interval_coverage": float(grouped.groupby("group").covered.all().mean()),
        "median_interval_width_z": float(np.median(hi - lo))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, exclusions = prepare()
    exclusions.to_csv(OUT / "excluded_labels.csv", index=False)
    frame.to_csv(OUT / "observed_features_and_labels.csv", index=False)
    train, cal, test = [frame.split == x for x in ("train", "calibration", "test")]
    for name, features in FAMILIES.items():
        if frame.loc[train, features].isna().all().any():
            missing = frame.loc[train, features].columns[frame.loc[train, features].isna().all()].tolist()
            raise ValueError(f"{name}: entirely missing training predictors {missing}; record availability before changing protocol")
    target = np.log1p(frame.spec_z.to_numpy())
    train_frame = frame.loc[train].reset_index(drop=True)
    train_target = target[train]
    cv = GroupKFold(3)
    cv_rows = []
    for name, features in FAMILIES.items():
        errors = []
        for fold, (fit, val) in enumerate(cv.split(train_frame, groups=train_frame.sky_group)):
            model = estimator()
            model.fit(train_frame.iloc[fit][features], train_target[fit])
            err = abs(model.predict(train_frame.iloc[val][features]) - train_target[val])
            group_error = pd.DataFrame({"group": train_frame.iloc[val].sky_group.to_numpy(), "error": err}).groupby("group").error.mean()
            errors.extend(group_error.tolist())
            cv_rows.append({"family": name, "fold": fold, "groups": len(group_error),
                            "group_mean_absolute_log1pz_error": float(group_error.mean())})
        cv_rows.append({"family": name, "fold": "all_group_means", "groups": len(errors),
                        "group_mean_absolute_log1pz_error": float(np.mean(errors))})
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(OUT / "training_group_cv.csv", index=False)
    totals = cv_table[cv_table.fold == "all_group_means"].set_index("family").group_mean_absolute_log1pz_error
    selected = sorted(totals.index, key=lambda k: (totals[k], k))[0]
    fitted, radii = {}, {}
    for name, features in FAMILIES.items():
        model = estimator()
        model.fit(frame.loc[train, features], target[train])
        cal_prediction = model.predict(frame.loc[cal, features])
        scores = pd.DataFrame({"group": frame.loc[cal, "sky_group"].to_numpy(),
            "error": abs(cal_prediction - target[cal])}).groupby("group").error.max()
        radius = conformal_radius(scores.to_numpy(), .1)
        if not np.isfinite(radius):
            raise ValueError("Insufficient calibration groups for a finite90% interval")
        fitted[name] = CalibratedRedshiftModel(model, features, radius,
            frame.loc[train, features].min().to_dict(), frame.loc[train, features].max().to_dict(),
            {"family": name, "feature_units": "ugriz observed nanomaggy-derived magnitudes; nativegrz and supplied angular descriptors",
             "target": "DR17 NSA_Z inherited heliocentric catalog redshift",
             "training_galaxies": int(train.sum()), "training_sky_groups": frame.loc[train, "sky_group"].nunique(),
             "calibration_galaxies": int(cal.sum()), "calibration_sky_groups": len(scores),
             "alpha": .1, "selection": "three-fold training sky-group CV only",
             "domain": "selected observed strong-spiral MaNGA/Legacy DR9; no external-survey validation",
             "arm_training_features": "sky-group out-of-fold predictions; human arm truth never a regressor input"})
        fitted[name].metadata["distance_arm_model_sha256"] = sha(ROOT / "executed_solid/segmentation/distance_feature_arm_model.joblib")
        fitted[name].metadata["cv_scope"] = "Training-group family selection conditional on precomputed OOF arm representation; not strictly nested end-to-end feature-learning CV. Final test groups remain unseen."
        radii[name] = radius
    median_center = float(np.median(target[train]))
    median_scores = pd.DataFrame({"group": frame.loc[cal, "sky_group"].to_numpy(),
        "error": abs(target[cal] - median_center)}).groupby("group").error.max()
    median_radius = conformal_radius(median_scores.to_numpy(), .1)
    frozen = {"deployed_family": selected, "training_cv_group_mean_error": totals.to_dict(),
        "feature_families": FAMILIES, "hyperparameters": {"trees": 128, "min_samples_leaf": 4,
        "max_features": 1., "seed": SEED}, "interval_alpha": .1,
        "group_max_conformal_radii_log1pz": radii, "test_labels_read_for_selection": False,
        "scientific_primary_contrast": "PGI versus PGIA, independent of deployment choice",
        "catalog_sha256": sha(DATA / "matched_catalog.csv"),
        "native_quality_manifest_sha256": sha(DATA / "native_images/manifest.json"),
        "descriptors_sha256": sha(ROOT / "executed_solid/segmentation/image_descriptors.csv"),
        "source_sha256": sha(__file__), "splits": {x: int((frame.split == x).sum()) for x in ("train", "calibration", "test")}}
    (OUT / "frozen_choice.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    # No prediction or score of a test galaxy is calculated before this point.
    results, output = {}, frame.loc[test, ["id", "sky_group", "spec_z", "drp3qual"]].copy()
    for name, model in fitted.items():
        result = model.predict(frame.loc[test])
        prediction, lo, hi = [result[key].to_numpy() for key in ("predicted_catalog_redshift", "redshift_interval_lower", "redshift_interval_upper")]
        results[name] = evaluate_predictions(frame.loc[test], prediction, lo, hi)
        results[name]["outside_training_range_galaxies"] = int(result.outside_training_feature_range.sum())
        for column in result:
            output[name + "_" + column] = result[column].to_numpy()
        model.save(OUT / "models" / name)
    med = np.full(test.sum(), np.expm1(median_center))
    med_lo = np.full(test.sum(), max(0., np.expm1(median_center - median_radius)))
    med_hi = np.full(test.sum(), np.expm1(median_center + median_radius))
    results["median"] = evaluate_predictions(frame.loc[test], med, med_lo, med_hi)
    output["median_prediction"] = med
    output.to_csv(OUT / "test_predictions.csv", index=False)
    chosen = fitted[selected]
    chosen.save(OUT / "deployed_model")
    ytest = frame.loc[test, "spec_z"].to_numpy()
    differences = paired_group_bootstrap(frame.loc[test, "sky_group"],
        abs(output.PGI_predicted_catalog_redshift.to_numpy() - ytest),
        abs(output.PGIA_predicted_catalog_redshift.to_numpy() - ytest), n_bootstrap=10000)
    subgroup_rows = []
    # Cut points from training only, not hand-picked after viewing test failures.
    cuts = np.quantile(frame.loc[train, "spec_z"], [1/3, 2/3])
    test_bins = np.digitize(ytest, cuts)
    for index in range(3):
        subset = test_bins == index
        if subset.any():
            for name in FAMILIES:
                pred = output[name + "_predicted_catalog_redshift"].to_numpy()[subset]
                subgroup_rows.append({"training_defined_redshift_bin": index, "family": name,
                    "galaxies": int(subset.sum()), "mae_z": float(np.mean(abs(pred-ytest[subset]))),
                    "signed_bias_z": float(np.mean(pred-ytest[subset]))})
    pd.DataFrame(subgroup_rows).to_csv(OUT / "redshift_subgroups.csv", index=False)
    try:
        conditional = cosmological_distances(output[selected + "_predicted_catalog_redshift"].to_numpy(),
            output[selected + "_redshift_interval_lower"].to_numpy(), output[selected + "_redshift_interval_upper"].to_numpy())
    except ValueError as exc:
        conditional = {"status": "optional_conversion_unavailable", "reason": str(exc),
                       "redshift_results_preserved": True}
    (OUT / "conditional_cosmological_conversion.json").write_text(json.dumps(conditional, indent=2) + "\n")
    summary = {"frozen_choice": frozen, "models": results, "paired_arm_increment_mae_z": differences,
        "test_spatial_groups": frame.loc[test, "sky_group"].nunique(),
        "label_exclusions": len(exclusions), "missing_image_rows_retained": int((frame.apparent_arm_features_available == 0).sum()),
        "train_defined_redshift_bin_edges": cuts.tolist(),
        "interpretation": "Statistical catalog-redshift experiment; arm contribution is PGI-versus-PGIA. A test ranking does not replace the training-selected deployment family.",
        "limits": ["160 selected nearby strong spirals, not an external survey or all-galaxy sample.",
            "Ground truth is inherited heliocentric catalog redshift, not independently measured geometric distance.",
            "Whole sky-tile calibration requires comparable exchangeable groups; no conditional/population-shift guarantee.",
            "Family-selection CV is conditional on cached OOF arm features, not a strictly nested end-to-end arm+redshift pipeline.",
            "MAE table weights galaxies equally; paired bootstrap weights sky-group mean errors equally.",
            "13 test groups support only imprecise comparative uncertainty.",
            "Human centers/star masks are supplied upstream; automated source detection is a different validation task.",
            "Photometry/noise/PSF nuisance features are shared to control seeing-depth correlations.",
            "Cosmological distance conversion is conditional and does not establish geometric-distance accuracy."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, style in (("P", "o"), ("PGI", "s"), ("PGIA", "^")):
        axes[0].scatter(ytest, output[name + "_predicted_catalog_redshift"], marker=style, label=name, alpha=.65)
    lim = max(ytest.max(), max(output[n + "_predicted_catalog_redshift"].max() for n in FAMILIES)) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=1); axes[0].set(xlabel="Catalog redshift", ylabel="Held-out prediction"); axes[0].legend()
    error_basic = abs(output.PGI_predicted_catalog_redshift.to_numpy() - ytest)
    error_arm = abs(output.PGIA_predicted_catalog_redshift.to_numpy() - ytest)
    axes[1].scatter(error_basic, error_arm); lim = max(error_basic.max(), error_arm.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=1)
    axes[1].set(xlabel="Absolute error: PGI", ylabel="Absolute error: PGIA (with arms)")
    chosen_prediction = output[selected + "_predicted_catalog_redshift"].to_numpy()
    order = np.argsort(ytest)
    lo = output[selected + "_redshift_interval_lower"].to_numpy()[order]
    hi = output[selected + "_redshift_interval_upper"].to_numpy()[order]
    axes[2].vlines(np.arange(len(order)), lo, hi, alpha=.55)
    axes[2].scatter(np.arange(len(order)), ytest[order], s=15, label="Catalog reference")
    axes[2].scatter(np.arange(len(order)), chosen_prediction[order], s=12, label=selected + " prediction")
    axes[2].set(xlabel="Held-out galaxy (sorted)", ylabel="Redshift and90% group-calibrated interval"); axes[2].legend(fontsize=8)
    fig.suptitle("New galaxies and sky groups; arm descriptors are predicted, not human truth")
    fig.savefig(OUT / "redshift_validation.png", dpi=160); plt.close(fig)
    print(json.dumps({"deployed_family": selected, "test_metrics": results,
                      "arm_increment": differences}, indent=2))


if __name__ == "__main__":
    main()
