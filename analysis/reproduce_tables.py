#!/usr/bin/env python3
"""Recompute grouped redshift tables and optionally refit the four controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def paired(frame, first, second, draws=100000):
    truth = frame.spec_z.to_numpy(float)
    raw = np.abs(frame[first].to_numpy(float)-truth) - np.abs(frame[second].to_numpy(float)-truth)
    groups = pd.Series(raw, index=frame.sky_group).groupby(level=0).mean().to_numpy()
    boot = np.random.default_rng(20261023).choice(groups, size=(draws, len(groups)), replace=True).mean(axis=1)
    return {"galaxies": len(frame), "sky_groups": len(groups),
            "equal_group_improvement_z": float(groups.mean()),
            "bootstrap95_z": np.quantile(boot, [.025, .975]).tolist(),
            "galaxy_mae_baseline_z": float(np.abs(frame[first]-frame.spec_z).mean()),
            "galaxy_mae_candidate_z": float(np.abs(frame[second]-frame.spec_z).mean()),
            "positive_means": "lower error after adding the candidate predictors",
            "scope": "Conditional on these fixed models, labels, groups and selected population"}


def refit(output):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    original = pd.read_csv(ROOT / "results/original_observed_features.csv")
    additional = pd.read_csv(ROOT / "results/additional_observed_features.csv")
    saved = pd.read_csv(ROOT / "results/controlled_predictions.csv")
    for frame in [original, additional]:
        frame["color_r_z"] = frame.mag_r - frame.mag_z
    train = original.loc[original.split == "train"].copy()
    test = additional.set_index("id").loc[saved.id]
    if len(train) != 91 or train.sky_group.nunique() != 38 or len(test) != 59:
        raise ValueError("Recorded cohort changed")
    if set(train.sky_group) & set(test.sky_group):
        raise ValueError("Training/evaluation sky groups overlap")
    reports = {}
    for family in ["L5", "L5A", "L3", "L3A"]:
        card = json.loads((ROOT / f"models/controls/{family}_model_card.json").read_text())
        if card["training_ids"] != train.id.tolist():
            raise ValueError("Training identity order changed")
        features = card["features"]
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4, max_features=1.,
                                random_state=20261023, n_jobs=1))
        model.fit(train[features], np.log1p(train.spec_z))
        prediction = np.maximum(0, np.expm1(model.predict(test[features])))
        if not np.allclose(prediction, saved[family], rtol=0, atol=1e-12):
            raise AssertionError(f"{family} does not reproduce saved predictions; check the recorded dependency versions")
        pd.DataFrame({"id": saved.id, "prediction": prediction}).to_csv(output / (family+"_refit.csv"), index=False)
        joblib.dump(model, output / (family+"_refit.joblib"), compress=3)
        reports[family] = {"train": len(train), "test": len(test),
                           "maximum_absolute_saved_prediction_difference": float(np.max(np.abs(prediction-saved[family])))}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refit-controls", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new or empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / "results/original_test_predictions.csv")
    additional = pd.read_csv(ROOT / "results/additional_predictions.csv")
    strong = additional.loc[additional.sample_group == "strong_spiral"]
    controls = pd.read_csv(ROOT / "results/controlled_predictions.csv")
    summary = {
        "original": paired(original, "PGI_predicted_catalog_redshift", "PGIA_predicted_catalog_redshift"),
        "additional_strong": paired(strong, "PGI_predicted_catalog_redshift", "PGIA_predicted_catalog_redshift"),
        "five_band_control": paired(controls, "L5", "L5A"),
        "three_band_control": paired(controls, "L3", "L3A"),
        "cohorts_pooled": False, "deployment_reselected": False,
    }
    if args.refit_controls:
        summary["control_refits"] = refit(args.output_dir)
    (args.output_dir / "reproduced_tables.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
