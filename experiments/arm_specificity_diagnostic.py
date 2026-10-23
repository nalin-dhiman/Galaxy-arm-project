#!/usr/bin/env python3
"""POST-HOC radial-confounding/rotation diagnosis of the frozen arm benchmark.

This script was proposed after primary held-out results were known. It changes
no primary result, filter, scale or threshold and cannot be a confirmatory test.
All image baselines are label-free. Human labels are used only for scoring.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.independent_filter_experiment import (load_object, map_labels,
    paired_interval, save_csv, sha)
from galaxy_lab.filtering import (radial_residual, refine_outer_geometry,
    DiskGeometry, evaluate_score)
from galaxy_lab.morphology import moment_geometry
OUT = ROOT / "executed_filters" / "specificity"
ANGLES = (45, 90, 135, 180, 225, 270, 315)


def metric(score, labels, valid):
    result = evaluate_score(score, labels, valid)
    ap, prevalence = result["average_precision"], result["positive_fraction"]
    result["AP_over_prevalence"] = ap / prevalence if ap is not None and prevalence else None
    return result


def rotated_votes(obj):
    """Rotate labels on the sky about the external human center, no light use."""
    annotation = obj["annotation"]
    im = obj["images"]["r"]
    yy, xx = np.indices(im.data.shape)
    target = im.wcs.pixel_to_world(xx, yy)
    with fits.open(ROOT / annotation["path"]) as hdus:
        awcs = WCS(hdus[annotation["wcs_hdu"]].header).celestial
        values = np.asarray(hdus[annotation["spiral_hdu"]].data, float)
        center = awcs.pixel_to_world(annotation["center_x"], annotation["center_y"])
        distance = center.separation(target)
        bearing = center.position_angle(target)
        rotations = {}
        for angle in ANGLES:
            # Inverse mapping: observed target pixels sample the unrotated mask
            # at bearing-angle. The same origin and angular radius are retained.
            source = center.directional_offset_by(bearing - angle * u.deg, distance)
            xs, ys = awcs.world_to_pixel(source)
            rotations[angle] = map_labels(values, xs, ys)
    return rotations


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / "executed_filters/benchmark/frozen_choice.json"
    frozen = json.loads(frozen_path.read_text())
    a_path = ROOT / "executed_filters/annotations/manifest.json"
    n_path = ROOT / "executed_filters/native_images/manifest.json"
    if sha(a_path) != frozen["annotation_manifest_sha256"] or sha(n_path) != frozen["native_manifest_sha256"]:
        raise ValueError("Source manifests differ from the primary frozen benchmark")
    a_data = json.loads(a_path.read_text())
    annotations = a_data.get("images", a_data.get("targets", []))
    n_data = json.loads(n_path.read_text())
    native = {r["id"]: r for r in n_data.get("targets", n_data.get("images", []))}
    selected = frozen["selected_pipeline"]
    if selected != "g/native_intensity":
        raise ValueError("This post-hoc diagnosis was specified for frozen g/native_intensity; no adaptive pipeline switching")
    test = [a for a in annotations if a["split"] == "test" and a["sample_group"] == "strong_spiral"]
    protocol = {
        "status": "POST-HOC EXPLORATORY DIAGNOSTIC AFTER PRIMARY TEST RESULTS WERE KNOWN",
        "selected_pipeline_unchanged": selected, "primary_choice_sha256": sha(frozen_path),
        "source_code_sha256": sha(__file__), "primary_binary_threshold_unchanged": frozen["binary_threshold_z"],
        "test_ids": [a["id"] for a in test], "no_filter_or_threshold_selection": True,
        "baselines": ["Image-only circular elliptical-radius median profile (zero inclination), smoothed radially with sigma=1 native pixel",
            "Image-only moment-seeded robust outer-isophote ellipse profile, single fixed attempt; failures reported, no best-baseline selection"],
        "rotation_sky_angles_deg": ANGLES,
        "rotation_mapping": "Inverse SkyCoord bearing rotation around external human center; nearest original human vote counts projected onto unresampled science grid",
        "evaluation_pixels": "primary common valid region intersected with label coverage for all fixed rotations, same pixels for every score and rotation on each galaxy",
        "label_threshold": frozen["label_votes"],
        "uncertainty": "10000 paired whole-galaxy bootstrap resamples; descriptive conditional intervals, no pixel bootstrap",
        "limitations": ["Post-hoc diagnostics cannot provide a fresh held-out confirmation or guide renewed selection on these 12 galaxies.",
            "A 180-degree mask rotation can realign a two-arm spiral; these rotations are not independent realizations or a formal null p-value.",
            "Mask rotation can alter prevalence on a masked region; AP, prevalence and AP/prevalence are all reported.",
            "Brightness may correlate with human arm annotation for radial and contrast reasons; AP alone does not identify spiral geometry.",
            "Image-only fitted ellipse is not independent kinematic geometry; its physical bias is unknown on these observations.",
            "Eight catalog controls without arm-task participation do not supply independent negative spatial truth and are omitted."]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    baseline_rows, rotation_rows, geometry_rows, failures = [], [], [], []
    for a in test:
        try:
            obj = load_object(a, native[a["id"]], psf_match=frozen["psf_match"])
            item = obj["maps"][selected]
            im = obj["images"]["g"]
            rotations = rotated_votes(obj)
            common = obj["common"].copy()
            for votes, covered in rotations.values():
                common &= covered
            truth = obj["votes"] >= frozen["label_votes"]
            actual = metric(item.score, truth, common)
            rr, _ = radial_residual(im, obj["geometry"])
            circular = metric(im.data - rr, truth, common)
            row = {"id": a["id"], "primary_valid_pixels": int(obj["common"].sum()),
                "diagnostic_valid_pixels": int(common.sum()),
                "coverage_retained": float(common.sum() / obj["common"].sum()),
                "positive_fraction": actual["positive_fraction"], "selected_AP": actual["average_precision"],
                "selected_AP_over_prevalence": actual["AP_over_prevalence"],
                "circular_profile_AP": circular["average_precision"],
                "circular_profile_AP_over_prevalence": circular["AP_over_prevalence"],
                "selected_minus_circular_AP": actual["average_precision"] - circular["average_precision"],
                "ellipse_profile_AP": None, "selected_minus_ellipse_AP": None}
            geometry_record = {"id": a["id"], "status": "not_attempted"}
            try:
                moment = moment_geometry(np.where(im.valid, im.data, np.nan), center_xy=obj["geometry"].center_xy)
                q = float(np.clip(moment["axis_ratio_moment"], .31, .99))
                initial = DiskGeometry(obj["geometry"].center_xy, moment["pa_image_deg_moment"],
                    float(np.degrees(np.arccos(q))), obj["geometry"].r_inner_arcsec,
                    obj["geometry"].r_outer_arcsec, "image-only moment seed, no human arm labels")
                fitted, diag = refine_outer_geometry(im, initial)
                er, _ = radial_residual(im, fitted)
                ellipse = metric(im.data - er, truth, common)
                row["ellipse_profile_AP"] = ellipse["average_precision"]
                row["selected_minus_ellipse_AP"] = actual["average_precision"] - ellipse["average_precision"]
                geometry_record = {"id": a["id"], "status": "fitted" if diag["success"] else "optimizer_unsuccessful",
                    "initial_PA_deg": initial.pa_deg, "initial_inclination_deg": initial.inclination_deg,
                    "fitted_PA_deg": fitted.pa_deg, "fitted_inclination_deg": fitted.inclination_deg,
                    "PA_change_deg_mod180": float((fitted.pa_deg - initial.pa_deg + 90) % 180 - 90),
                    "inclination_change_deg": fitted.inclination_deg - initial.inclination_deg, **diag,
                    "PA_search_bound_hit": bool(abs((fitted.pa_deg - initial.pa_deg + 90) % 180 - 90) > 44.9),
                    "axis_ratio_bound_hit": bool(diag["axis_ratio"] < .3001 or diag["axis_ratio"] > .9999),
                    "physical_bias_measurable": False}
            except Exception as exc:
                geometry_record = {"id": a["id"], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            geometry_rows.append(geometry_record)
            local_rotation = []
            for angle, (votes, _) in rotations.items():
                result = metric(item.score, votes >= frozen["label_votes"], common)
                rotation_rows.append({"id": a["id"], "sky_angle_deg": angle, **result})
                local_rotation.append(result)
            aps = [r["average_precision"] for r in local_rotation]
            ratios = [r["AP_over_prevalence"] for r in local_rotation]
            if any(v is None for v in aps + ratios):
                row["mean_rotated_AP"] = row["selected_minus_mean_rotated_AP"] = None
                row["mean_rotated_AP_over_prevalence"] = row["selected_minus_mean_rotated_AP_over_prevalence"] = None
                row["rotation_status"] = "at_least_one_fixed_rotation_lacks_both_classes; no angle selection"
            else:
                row["mean_rotated_AP"] = float(np.mean(aps))
                row["selected_minus_mean_rotated_AP"] = actual["average_precision"] - np.mean(aps)
                row["mean_rotated_AP_over_prevalence"] = float(np.mean(ratios))
                row["selected_minus_mean_rotated_AP_over_prevalence"] = actual["AP_over_prevalence"] - np.mean(ratios)
                row["rotation_status"] = "all_seven_fixed_angles_scored"
            baseline_rows.append(row)
            print(a["id"], "actual", row["selected_AP"], "circular", row["circular_profile_AP"], "rotated_mean", row["mean_rotated_AP"], flush=True)
        except Exception as exc:
            failures.append({"id": a["id"], "reason": f"{type(exc).__name__}: {exc}"})
    save_csv(OUT / "galaxy_baselines.csv", baseline_rows)
    save_csv(OUT / "all_mask_rotations.csv", rotation_rows)
    (OUT / "ellipse_fit_diagnostics.json").write_text(json.dumps(geometry_rows, indent=2, allow_nan=False) + "\n")
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    def values(key):
        return [float(r[key]) for r in baseline_rows if r.get(key) is not None]
    summary = {"status": protocol["status"], "frozen_pipeline": selected,
        "requested_galaxies": len(test), "scored_galaxies": len(baseline_rows), "failures": failures,
        "primary_results_changed": False, "selection_or_threshold_retuned": False,
        "macro_selected_AP": float(np.mean(values("selected_AP"))),
        "macro_circular_profile_AP": float(np.mean(values("circular_profile_AP"))),
        "macro_ellipse_profile_AP_defined_subset": float(np.mean(values("ellipse_profile_AP"))) if values("ellipse_profile_AP") else None,
        "ellipse_profile_defined_galaxies": len(values("ellipse_profile_AP")),
        "ellipse_PA_search_bound_hit_galaxies": sum(bool(r.get("PA_search_bound_hit")) for r in geometry_rows),
        "ellipse_axis_ratio_bound_hit_galaxies": sum(bool(r.get("axis_ratio_bound_hit")) for r in geometry_rows),
        "ellipse_max_abs_inclination_change_deg": float(max(abs(r.get("inclination_change_deg", 0)) for r in geometry_rows)) if geometry_rows else None,
        "macro_mean_rotated_AP": float(np.mean(values("mean_rotated_AP"))) if values("mean_rotated_AP") else None,
        "selected_minus_circular_AP": paired_interval(values("selected_minus_circular_AP")),
        "selected_minus_ellipse_AP_defined_subset": paired_interval(values("selected_minus_ellipse_AP")),
        "selected_minus_mean_rotated_AP": paired_interval(values("selected_minus_mean_rotated_AP")),
        "selected_minus_mean_rotated_AP_over_prevalence": paired_interval(values("selected_minus_mean_rotated_AP_over_prevalence")),
        "minimum_common_coverage_retained": min(values("coverage_retained")) if baseline_rows else None,
        "rotation_macro_AP_by_angle": {str(a): float(np.mean([r["average_precision"] for r in rotation_rows if r["sky_angle_deg"] == a and r["average_precision"] is not None])) for a in ANGLES},
        "interpretation": "POST-HOC confounding check only. High pixel AP can reflect radial brightness and galaxy orientation rather than successful arm tracing. No fresh held-out claim, formal rotation p-value, distance or viscosity inference."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    actual = np.array(values("selected_AP")); circular = np.array(values("circular_profile_AP"))
    axes[0].scatter(circular, actual, color="tab:blue", label="12 test galaxies")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Circular radial-profile AP", ylabel="Frozen g-brightness AP", title="Radial light can explain mask overlap")
    for index, row in enumerate(baseline_rows):
        axes[1].plot([0, 1], [row["mean_rotated_AP"], row["selected_AP"]], "o-", alpha=.7)
    axes[1].set(xticks=[0, 1], xticklabels=["Mean seven rotations", "Actual alignment"], ylabel="Pixel AP", ylim=(0, 1), title="Fixed rotations; 180° can align m=2")
    means = [summary["rotation_macro_AP_by_angle"][str(a)] for a in ANGLES]
    axes[2].plot(ANGLES, means, "o-", label="Rotated human masks")
    axes[2].axhline(summary["macro_selected_AP"], color="black", ls="--", label="Actual human masks")
    axes[2].set(xlabel="Fixed mask sky rotation [deg]", ylabel="Macro pixel AP", title="No angle selection or null p-value")
    axes[2].legend(fontsize=8)
    fig.suptitle("POST-HOC diagnostic after primary test results: no new confirmatory evidence", fontsize=11)
    fig.savefig(OUT / "radial_rotation_diagnostic.png", dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
