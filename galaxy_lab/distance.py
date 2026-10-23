"""Group-disjoint, supervised image-feature distance/redshift evaluation.

This evaluates calibrated statistical predictions; it supplies no physical
standard ruler.  Inputs must be calibrated image-only observables and an
independent label.  Column-name checks reduce accidental leakage but cannot
prove feature provenance.  Redshift predictions are not converted into distance.

Split-conformal intervals use one score per calibration GROUP (the maximum
residual within each group), rather than treating related galaxies as iid.
Coverage concerns a new exchangeable group with comparable selection and group
construction.  It does not promise conditional coverage or validity under survey
or population shift.  Reference: https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


_FEATURE_PREFIXES = (
    "angular_", "apparent_", "mag_", "color_", "flux_", "surface_brightness_",
    "axis_ratio", "concentration", "asymmetry", "pitch_", "arm_", "fourier_",
    "ellipticity", "sersic_", "pa_", "inclination_", "snr_", "psf_",
    "petro_", "iso_", "fwhm_", "gini", "m20", "clumpiness", "entropy",
)
_FORBIDDEN = re.compile(
    r"(^|_)(distance|dist|redshift|spec|label|target|mass|stellar|luminosity|"
    r"absolute|absmag|abs_mag|physical|kpc\d*|mpc\d*|pc\d*|msun|lsun|halo|rho|density|sigma|velocity|rotation|"
    r"viscosity|viscous|fluid|dynamical|dynamics|id|ra|dec)(_|$)", re.I
)


def validate_feature_names(photometric_features, morphology_features):
    """Require explicit observable names and reject common target-derived data.

    Use e.g. apparent_mag_r, color_g_r, angular_radius_arcsec, pitch_deg,
    arm_spacing_pixels.  Physical sizes, absolute magnitudes, luminosities,
    dynamics, identifiers and redshift/distance columns are forbidden.
    """
    photo, morph = list(photometric_features), list(morphology_features)
    if not photo or not morph:
        raise ValueError("both photometric_features and morphology_features are required")
    names = photo + morph
    if len(set(names)) != len(names):
        raise ValueError("feature lists must contain distinct nonoverlapping columns")
    for name in names:
        if not isinstance(name, str) or _FORBIDDEN.search(name):
            raise ValueError(f"forbidden or potentially label-derived predictor: {name!r}")
        if not name.lower().startswith(_FEATURE_PREFIXES):
            raise ValueError(f"predictor {name!r} needs an explicit image-observable name")
    return photo, morph


def group_disjoint_split(groups, seed=20261023, calibration_fraction=.2, test_fraction=.2):
    """Assign whole groups, deterministically, to train/calibration/test."""
    if not (0 < calibration_fraction < 1 and 0 < test_fraction < 1 and
            calibration_fraction + test_fraction < 1):
        raise ValueError("split fractions must be positive and sum to less than one")
    group_array = np.asarray(groups).astype(str)
    unique = np.unique(group_array)
    if len(unique) < 5:
        raise ValueError("at least five independent groups are required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    ncal = max(1, int(np.floor(len(unique)*calibration_fraction)))
    ntest = max(1, int(np.floor(len(unique)*test_fraction)))
    lookup = {g: "calibration" for g in shuffled[:ncal]}
    lookup.update({g: "test" for g in shuffled[ncal:ncal+ntest]})
    lookup.update({g: "train" for g in shuffled[ncal+ntest:]})
    return np.array([lookup[g] for g in group_array])


def conformal_radius(scores, alpha=.1):
    """Finite-sample split-conformal order statistic, including infinity.

    With n independent calibration scores use k=ceil((n+1)(1-alpha)).
    If k>n, the requested coverage requires an infinite interval.  This must
    not be replaced by clipping the quantile level to one.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("scores must be a nonempty finite nonnegative vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(math.ceil((len(values)+1)*(1-alpha)))
    return float("inf") if rank > len(values) else float(np.partition(values, rank-1)[rank-1])


def paired_group_bootstrap(groups, baseline_losses, candidate_losses, seed=20261023, n_bootstrap=2000):
    """CI for baseline-minus-candidate error, weighting independent groups equally."""
    if n_bootstrap < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    frame = pd.DataFrame({"group": np.asarray(groups).astype(str),
                          "improvement": np.asarray(baseline_losses)-np.asarray(candidate_losses)})
    group_delta = frame.groupby("group", sort=True).improvement.mean().to_numpy()
    if len(group_delta)==0:
        raise ValueError("at least one test group is required")
    if len(group_delta)==1:
        return {"mean_improvement":float(group_delta[0]),"ci95_low":None,"ci95_high":None,
                "n_test_groups":1,"bootstrap_draws":0,"resampling_unit":"group",
                "positive_means":"candidate has lower error",
                "status":"insufficient independent test groups for a bootstrap confidence interval",
                "limitation":"one test group cannot estimate between-group sampling uncertainty"}
    rng = np.random.default_rng(seed)
    # Loop avoids an unnecessarily large n_bootstrap-by-n_groups matrix.
    samples = np.array([np.mean(rng.choice(group_delta, len(group_delta), replace=True))
                        for _ in range(n_bootstrap)])
    return {"mean_improvement": float(group_delta.mean()),
            "ci95_low": float(np.quantile(samples,.025)),
            "ci95_high": float(np.quantile(samples,.975)),
            "n_test_groups": len(group_delta), "bootstrap_draws": n_bootstrap,
            "resampling_unit": "group", "positive_means": "candidate has lower error",
            "limitation": "conditional on this trained model and split; no retraining uncertainty"}


def evaluate_distance_features(frame, photometric_features, morphology_features, *,
                               target="distance_mpc", seed=20261023, alpha=.1,
                               n_estimators=128, n_bootstrap=2000,
                               synthetic=False, provenance=None, missing_indicators=True):
    """Compare median, photometry, and photometry+morphology on identical test rows.

    Returns ``(summary_dict, predictions_dataframe, split_manifest_dataframe)``.
    Required columns: galaxy_id, group_id, and either distance_mpc (>0) or
    spec_z (>=0).  Every galaxy must have exactly one feature row.  Related
    galaxies/views or survey regions must share group_id.  Groups are split
    60/20/20 before imputation, fitting, or interval calibration.  Fixed model
    hyperparameters are not tuned on the test or calibration partition.
    """
    photo, morph = validate_feature_names(photometric_features, morphology_features)
    if target not in ("distance_mpc", "spec_z"):
        raise ValueError("target must be distance_mpc or spec_z; no implicit cosmology conversion")
    required = ["galaxy_id", "group_id", target] + photo + morph
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    data = frame.loc[:, required].copy().reset_index(drop=True)
    if data[["galaxy_id","group_id"]].isna().any().any():
        raise ValueError("identifiers and group IDs cannot be missing")
    if data.galaxy_id.astype(str).duplicated().any():
        raise ValueError("one row per galaxy_id required; merge repeated-image features first")
    y = pd.to_numeric(data[target], errors="raise").to_numpy(float)
    if np.any(~np.isfinite(y)) or np.any(y <= 0 if target=="distance_mpc" else y < 0):
        raise ValueError("target labels must be finite, with distance>0 or spec_z>=0")
    for name in photo+morph:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if np.isinf(data[name].to_numpy(float)).any():
            raise ValueError(f"infinite values are invalid for predictor {name}")
    split = group_disjoint_split(data.group_id, seed)
    train, cal, test = (split == name for name in ("train","calibration","test"))
    for name in photo+morph:
        if data.loc[train,name].isna().all():
            raise ValueError(f"predictor {name!r} is entirely missing in training")
    transform = np.log if target=="distance_mpc" else np.log1p
    inverse = np.exp if target=="distance_mpc" else np.expm1
    transformed = transform(y)
    test_data = data.loc[test]
    output = test_data[["galaxy_id","group_id",target]].copy()
    summary = {
        "experiment_kind": "SYNTHETIC DEMONSTRATION ONLY" if synthetic else "observational label evaluation",
        "target": target, "target_transform": "ln(distance_mpc)" if target=="distance_mpc" else "ln(1+spec_z)",
        "distance_from_redshift_conversion": "none",
        "seed": seed, "alpha": alpha, "feature_lists": {"photometry":photo,"morphology":morph},
        "model_configuration": {"estimator":"ExtraTreesRegressor", "n_estimators":n_estimators,
                                "min_samples_leaf":3,"max_features":1.0,"n_jobs":1,
                                "imputation":"training median with missingness indicators" if missing_indicators else "training median; explicit availability flags only",
                                "hyperparameter_selection":"fixed before all splits; no tuning"},
        "software_versions": {name:version(name) for name in ("numpy","pandas","scikit-learn")},
        "split_rows": {name:int(np.sum(split==name)) for name in ("train","calibration","test")},
        "split_groups": {name:int(data.loc[split==name,"group_id"].nunique()) for name in ("train","calibration","test")},
        "content_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "provenance": provenance or {}, "models": {},
        "coverage_assumption": "exchangeable new groups, comparable selection, no survey/population shift",
        "conformal_score": "maximum absolute transformed-label residual within each calibration group",
        "limits": ["column names cannot establish independent feature provenance",
                   "label errors and selection effects are not deconvolved",
                   "no conditional or shifted-population coverage guarantee",
                   "synthetic results never establish observational distance accuracy"],
    }
    predictions_transformed = {}
    feature_sets = {"median": [], "photometry": photo, "photometry_morphology":photo+morph}
    for name, features in feature_sets.items():
        if not features:
            prediction_cal = np.full(cal.sum(), np.median(transformed[train]))
            prediction_test = np.full(test.sum(), np.median(transformed[train]))
            imputation = {}
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median",add_indicator=missing_indicators),
                ExtraTreesRegressor(n_estimators=n_estimators,min_samples_leaf=3,
                                    max_features=1.0,n_jobs=1,random_state=seed),
            )
            model.fit(data.loc[train,features],transformed[train])
            prediction_cal = model.predict(data.loc[cal,features])
            prediction_test = model.predict(data.loc[test,features])
            imputation = dict(zip(features,model[0].statistics_.tolist()))
        predictions_transformed[name] = prediction_test
        scores = pd.DataFrame({"group":data.loc[cal,"group_id"].astype(str).to_numpy(),
                               "score":np.abs(transformed[cal]-prediction_cal)}).groupby("group").score.max()
        radius = conformal_radius(scores.to_numpy(),alpha)
        predicted = inverse(prediction_test)
        with np.errstate(over="ignore"):
            lower = np.maximum(0, inverse(prediction_test-radius))
            upper = inverse(prediction_test+radius)
        covered = (y[test] >= lower) & (y[test] <= upper)
        native_error = np.abs(predicted-y[test])
        transformed_error = np.abs(prediction_test-transformed[test])
        group_cover = pd.DataFrame({"g":test_data.group_id.astype(str).to_numpy(),"covered":covered}).groupby("g").covered.all()
        record = {
            "mae_native_target_units":float(native_error.mean()),
            "mean_signed_error_native_target_units":float(np.mean(predicted-y[test])),
            "median_absolute_error_native":float(np.median(native_error)),
            "mae_transformed_target":float(transformed_error.mean()),
            "group_mean_mae_transformed_target":float(pd.DataFrame({"g":test_data.group_id.astype(str).to_numpy(),"e":transformed_error}).groupby("g").e.mean().mean()),
            "galaxy_interval_coverage":float(covered.mean()),
            "whole_group_interval_coverage":float(group_cover.mean()),
            "median_interval_width_native":float(np.median(upper-lower)),
            "conformal_radius_transformed":radius,
            "calibration_groups":len(scores), "train_only_imputation_medians":imputation,
        }
        if target=="distance_mpc":
            record["median_absolute_fractional_distance_error"] = float(np.median(native_error/y[test]))
            record["mean_signed_fractional_distance_error"] = float(np.mean((predicted-y[test])/y[test]))
        else:
            dz = (predicted-y[test])/(1+y[test])
            record["mean_signed_delta_z_over_1plusz"] = float(np.mean(dz))
            record["sigma_nmad_delta_z_over_1plusz"] = float(1.48*np.median(np.abs(dz-np.median(dz))))
            record["outlier_fraction_abs_delta_z_over_1plusz_gt_0p15"] = float(np.mean(np.abs(dz)>.15))
        summary["models"][name] = record
        output[f"{name}_prediction"] = predicted
        output[f"{name}_lower"] = lower
        output[f"{name}_upper"] = upper
    summary["paired_morphology_improvement"] = paired_group_bootstrap(
        test_data.group_id, np.abs(predictions_transformed["photometry"]-transformed[test]),
        np.abs(predictions_transformed["photometry_morphology"]-transformed[test]),
        seed=seed,n_bootstrap=n_bootstrap,
    )
    summary["paired_morphology_improvement"]["error_units"] = summary["target_transform"]
    manifest = data[["galaxy_id","group_id"]].copy()
    manifest["split"] = split
    return summary, output.reset_index(drop=True), manifest


def synthetic_distance_demo(n_groups=180, galaxies_per_group=2, seed=20261023):
    """Engineered recovery demonstration, with NO observational realism claim."""
    rng = np.random.default_rng(seed)
    n = n_groups*galaxies_per_group
    a,b = rng.normal(size=(2,n))
    group_effect = np.repeat(rng.normal(0,.04,n_groups),galaxies_per_group)
    log_distance = 3.6 + .65*a + .6*b + group_effect + rng.normal(0,.035,n)
    return pd.DataFrame({"galaxy_id":[f"synthetic_{i}" for i in range(n)],
                         "group_id":np.repeat([f"synthetic_group_{i}" for i in range(n_groups)],galaxies_per_group),
                         "distance_mpc":np.exp(log_distance),
                         "apparent_mag_r":18+1.4*a,
                         "color_g_r":.5+.1*a+rng.normal(0,.03,n),
                         "pitch_deg":20+5*b,
                         "arm_contrast":rng.uniform(.1,.8,n)})


def _json_safe(value):
    """Encode infinite interval endpoints explicitly without invalid JSON NaN."""
    if isinstance(value,dict):
        return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return "+infinity" if value>0 else "-infinity"
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv",type=Path)
    source.add_argument("--synthetic-demo",action="store_true")
    parser.add_argument("--photometric",nargs="+")
    parser.add_argument("--morphology",nargs="+")
    parser.add_argument("--target",choices=["distance_mpc","spec_z"],default="distance_mpc")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--alpha",type=float,default=.1,
                        help="miscoverage probability; default .1 gives nominal 90%% intervals")
    args = parser.parse_args(argv)
    if args.synthetic_demo:
        data = synthetic_distance_demo(seed=args.seed)
        photo,morph = ["apparent_mag_r","color_g_r"],["pitch_deg","arm_contrast"]
        provenance = {"source":"engineered synthetic distribution; not real galaxy images"}
        if args.target!="distance_mpc":
            parser.error("the synthetic demonstration has distance labels only")
    else:
        if not args.photometric or not args.morphology:
            parser.error("observational CSV requires explicit --photometric and --morphology lists")
        data = pd.read_csv(args.csv,dtype={"galaxy_id":str,"group_id":str})
        photo,morph = args.photometric,args.morphology
        provenance = {"input_csv":str(args.csv.resolve()),"raw_sha256":hashlib.sha256(args.csv.read_bytes()).hexdigest()}
    summary,predictions,manifest = evaluate_distance_features(
        data,photo,morph,target=args.target,seed=args.seed,alpha=args.alpha,
        synthetic=args.synthetic_demo,provenance=provenance)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"distance_summary.json").write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False)+"\n")
    predictions.to_csv(args.out/"distance_test_predictions.csv",index=False)
    manifest.to_csv(args.out/"distance_split_manifest.csv",index=False)
    print(json.dumps(_json_safe({"experiment_kind":summary["experiment_kind"],
                                "models":summary["models"],
                                "paired_morphology_improvement":summary["paired_morphology_improvement"]}),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
