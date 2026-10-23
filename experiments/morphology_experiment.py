#!/usr/bin/env python3
"""Execute injected validation and available real FITS analysis.

Run from repository root: OPENBLAS_NUM_THREADS=1 python experiments/morphology_experiment.py
Optional real inputs: executed_research/images/manifest.json, schema documented
in executed_research/morphology/README.txt after running. No network is required.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.morphology import (load_fits, measure_image, sky_center_and_pa,
                                   synthetic_spiral)

OUT = ROOT / "executed_research" / "morphology"


def public_json(result):
    if isinstance(result, dict):
        return {key: public_json(value) for key, value in result.items()
                if key not in ("diagnostics", "wcs")}
    if isinstance(result, (list, tuple)):
        return [public_json(v) for v in result]
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, np.generic):
        return result.item()
    return result


def save_json(path, value):
    path.write_text(json.dumps(public_json(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostic(image, result, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    # This asinh transform is confined to the plotting stage.
    clean = image - result["background"]
    finite = clean[np.isfinite(clean)]
    scale = max(float(np.percentile(finite, 95)) / 4, 1e-8)
    axes[0].imshow(np.arcsinh(clean / scale), origin="lower", cmap="gray", vmin=-.1,
                   vmax=np.arcsinh(max(np.percentile(finite, 99), scale) / scale))
    cx, cy = result["center_x_px"], result["center_y_px"]
    pa, inc = np.radians(result["pa_image_deg_used"]), np.radians(result["inclination_deg_used"])
    theta = np.linspace(0, 2 * np.pi, 300)
    for radius in (result["r_inner_px"], result["r_outer_px"]):
        X, Y = radius * np.cos(theta), radius * np.sin(theta) * np.cos(inc)
        axes[0].plot(cx + np.cos(pa) * X - np.sin(pa) * Y,
                     cy + np.sin(pa) * X + np.cos(pa) * Y, color="tomato", lw=1)
    axes[0].plot(cx, cy, "+", color="cyan")
    axes[0].set_title("Native image; display asinh only")
    diag = result["spiral"]["diagnostics"]
    axes[1].imshow(diag["polar_relative"], origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-.7, vmax=.7, extent=(0, 360, diag["log_radius"][0], diag["log_radius"][-1]))
    axes[1].set(xlabel="Deprojected azimuth [deg]", ylabel="ln radius [pixels]", title="Relative light residual in annulus")
    for mode_index, (mode, amplitude) in enumerate(diag["spectra"].items()):
        # Split opposite winding senses to avoid drawing across the excluded gap.
        p = diag["pitch_grid_deg"]
        for mask in (p < 0, p > 0):
            axes[2].plot(p[mask], amplitude[mask], color=plt.get_cmap("tab10")(mode_index),
                         label=f"m={mode}" if (p[mask] > 0).all() else None)
    axes[2].set(xlabel="Signed logarithmic pitch [deg]", ylabel="Relative Fourier amplitude", title="Mode spectrum; m means periodicity")
    axes[2].legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def synthetic_validation():
    rows, all_results = [], []
    # Variation across pitch, mode, geometry, blur and noise. Geometry supplied
    # separately from pixels: these cases test measurement, not geometry recovery.
    cases = [
        (12, 2, 0, 10, .8, .01), (22, 2, 35, 25, 1., .02),
        (32, 3, 55, 110, 1.5, .04), (42, 4, 30, 77, 1.5, .025),
        (18, 2, 65, 60, 2., .04), (30, 2, 20, 150, 3., .06),
        (10, 3, 45, 0, 1.5, .03), (35, 4, 60, 95, 2., .04),
    ]
    for index, (pitch, mode, inc, pa, psf, noise) in enumerate(cases):
        image, truth = synthetic_spiral(pitch_deg=pitch, mode_m=mode, inclination_deg=inc,
            pa_deg=pa, psf_sigma_px=psf, noise_sigma=noise, seed=90 + index)
        for condition in ("supplied_geometry", "automatic_moments", "resampled_0.65"):
            args = {k: v for k, v in truth.items() if k not in ("pitch_deg", "mode_m")}
            working = image
            if condition == "automatic_moments":
                args.pop("pa_deg")
                args.pop("inclination_deg")
            if condition == "resampled_0.65":
                working = zoom(image, .65, order=1)
                args["center_xy"] = ((working.shape[1] - 1) / 2, (working.shape[0] - 1) / 2)
                args["r_inner_px"] *= .65
                args["r_outer_px"] *= .65
            key = f"synthetic_{index:02d}_{condition}"
            row = {"id": key, "condition": condition, "pitch_truth_deg": pitch,
                   "mode_truth": mode, "inclination_truth_deg": inc, "pa_truth_deg": pa,
                   "psf_sigma_original_px": psf, "noise_sigma": noise}
            try:
                result = measure_image(working, **args, sensitivity=True,
                                       pixel_scale_arcsec=.4 / (.65 if condition == "resampled_0.65" else 1))
                spiral = result["spiral"]
                row.update(status=result["status"], pitch_measured_deg=spiral["pitch_abs_deg"],
                    pitch_error_deg=spiral["pitch_abs_deg"] - pitch,
                    mode_measured=spiral["mode_m"], mode_correct=spiral["mode_m"] == mode,
                    inclination_used_deg=result["inclination_deg_used"],
                    flags=" | ".join(result["flags"]))
                if index < 2 and condition != "resampled_0.65":
                    plot_diagnostic(working, result, OUT / f"{key}.png", key)
                all_results.append({"id": key, "truth": truth, "measurements": result})
            except Exception as error:
                row.update(status="failed", flags=str(error))
            rows.append(row)
    write_csv(OUT / "synthetic_validation.csv", rows)
    save_json(OUT / "synthetic_measurements.json", all_results)
    summary = {}
    for condition in ("supplied_geometry", "automatic_moments", "resampled_0.65"):
        sub = [r for r in rows if r["condition"] == condition]
        success = [r for r in sub if "pitch_error_deg" in r]
        correct = [r for r in success if r["mode_correct"]]
        summary[condition] = {"cases": len(sub), "measurement_completed": len(success),
            "mode_correct": len(correct),
            "median_abs_pitch_error_deg_all_completed": float(np.median([abs(r["pitch_error_deg"]) for r in success])) if success else None,
            "max_abs_pitch_error_deg_all_completed": float(max(abs(r["pitch_error_deg"]) for r in success)) if success else None}
    # Show the complete pipeline's failure on a null field explicitly.
    yy, xx = np.indices((256, 256))
    null = .2 + np.exp(-np.hypot(xx - 127.5, yy - 127.5) / 45)
    null += np.random.default_rng(42).normal(0, .02, null.shape)
    null_result = measure_image(null, center_xy=(127.5, 127.5), pa_deg=0,
                                inclination_deg=0, r_inner_px=28, r_outer_px=88)
    summary["axisymmetric_null"] = {"amplitude": null_result["spiral"]["amplitude"],
                                     "flags": null_result["flags"]}
    save_json(OUT / "axisymmetric_null.json", null_result)
    save_json(OUT / "synthetic_summary.json", summary)
    # Comparison figure makes the geometry caveat visible beside nominal tests.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for condition, marker in [("supplied_geometry", "o"), ("automatic_moments", "x"), ("resampled_0.65", "+")]:
        sub = [r for r in rows if r["condition"] == condition and "pitch_measured_deg" in r]
        ax.scatter([r["pitch_truth_deg"] for r in sub], [r["pitch_measured_deg"] for r in sub], marker=marker, label=condition.replace("_", " "))
    ax.plot([0, 60], [0, 60], color="gray", lw=1)
    ax.set(xlabel="Injected pitch [deg]", ylabel="Measured dominant-mode pitch [deg]",
           title="Synthetic recovery: known geometry versus automatic moment geometry", xlim=(0, 60), ylim=(0, 60))
    ax.legend()
    fig.savefig(OUT / "synthetic_recovery.png", dpi=160)
    plt.close(fig)
    return summary


def real_measurements(manifest_path):
    if not manifest_path.exists():
        return {"status": "not_run", "reason": "No downloaded FITS manifest exists"}
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["images"] if isinstance(manifest, dict) else manifest
    results, rows = [], []
    for entry in entries:
        identifier = entry.get("id", Path(entry["path"]).stem)
        identifier = "".join(c if c.isalnum() or c in "_-" else "_" for c in identifier)
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
            if not path.exists():
                path = manifest_path.parent / entry["path"]
        try:
            loaded = load_fits(path, entry.get("hdu"))
            params = {k: entry[k] for k in ("center_xy", "pa_deg", "inclination_deg", "intrinsic_thickness",
                "r_inner_px", "r_outer_px", "distance_mpc") if k in entry}
            if "ra_deg" in entry and "dec_deg" in entry:
                center, sky_pa = sky_center_and_pa(loaded.metadata, entry["ra_deg"], entry["dec_deg"], entry.get("pa_sky_deg"))
                params.setdefault("center_xy", center)
                if sky_pa is not None:
                    params["pa_deg"] = sky_pa
            params["pixel_scale_arcsec"] = entry.get("pixel_scale_arcsec", loaded.metadata.get("pixel_scale_arcsec"))
            result = measure_image(loaded.data, **params)
            result["input"] = entry
            result["fits_metadata"] = public_json(loaded.metadata)
            # Real images lack measured pitch ground truth. Always distinguish
            # a produced descriptor from a validated science measurement.
            result["flags"].append("REAL_IMAGE_DESCRIPTOR_NOT_VALIDATED_AGAINST_EXTERNAL_ARM_TRACING")
            result["status"] = "exploratory_measurement"
            save_json(OUT / f"{identifier}.json", result)
            plot_diagnostic(loaded.data, result, OUT / f"{identifier}.png", identifier)
            row = {"id": identifier, "status": result["status"],
                   "band": entry.get("band", loaded.metadata["filter"]),
                   "pixel_scale_arcsec": params["pixel_scale_arcsec"],
                   "pitch_abs_deg": result["spiral"]["pitch_abs_deg"], "mode_m": result["spiral"]["mode_m"],
                   "inclination_deg": result["inclination_deg_used"],
                   "r50_aperture_arcsec": result.get("r50_aperture_arcsec"),
                   "r90_aperture_arcsec": result.get("r90_aperture_arcsec"),
                   "pitch_sensitivity_min_deg": result["spiral"].get("sensitivity_pitch_min_deg"),
                   "pitch_sensitivity_max_deg": result["spiral"].get("sensitivity_pitch_max_deg"),
                   "distance_mpc_external": entry.get("distance_mpc"),
                   "flags": " | ".join(result["flags"])}
            results.append({"id": identifier, "status": "exploratory_measurement"})
        except Exception as error:
            row = {"id": identifier, "status": "failed", "flags": f"{type(error).__name__}: {error}"}
            results.append(row)
            save_json(OUT / f"{identifier}.json", {"input": entry, **row})
        rows.append(row)
    if rows:
        write_csv(OUT / "real_measurements.csv", rows)
    summary = {"status": "executed", "image_count": len(rows),
               "completed": sum(r["status"] != "failed" for r in rows), "results": results}
    save_json(OUT / "real_summary.json", summary)
    return summary


def m51_crossband_comparison(manifest_path):
    """Repeat g/r/z with the same r-derived geometry and angular annulus."""
    if not manifest_path.exists() or not (OUT / "M51_r.json").exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    selected = {item["band"]: item for item in manifest["images"]
                if item.get("id", "").startswith("M51_")}
    if len(selected) < 2:
        return None
    baseline = json.loads((OUT / "M51_r.json").read_text())
    if "center_x_px" not in baseline:
        return None
    common = {"pa_deg": baseline["pa_image_deg_used"],
              "inclination_deg": baseline["inclination_deg_used"],
              "r_inner_px": baseline["r_inner_px"], "r_outer_px": baseline["r_outer_px"],
              "sensitivity": False}
    rows = []
    fig, axes = plt.subplots(1, len(selected), figsize=(4.6 * len(selected), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (band, entry) in zip(axes, sorted(selected.items())):
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        loaded = load_fits(path, entry.get("hdu"))
        center, _ = sky_center_and_pa(loaded.metadata, entry["ra_deg"], entry["dec_deg"])
        result = measure_image(loaded.data, center_xy=center, **common)
        spiral = result["spiral"]
        modes = {row["mode_m"]: row for row in spiral["modes"]}
        ordered = sorted(spiral["modes"], key=lambda row: row["amplitude"], reverse=True)
        ratio = ordered[0]["amplitude"] / max(ordered[1]["amplitude"], 1e-12)
        flags = list(result["flags"])
        if ratio < 1.25:
            flags.append("COMPETING_FOURIER_MODES_WITHIN_25PCT")
        rows.append({"band": band, "dominant_mode": spiral["mode_m"],
            "dominant_candidate_pitch_deg": spiral["pitch_abs_deg"],
            "m2_candidate_pitch_deg": modes[2]["pitch_abs_deg"],
            "m2_amplitude": modes[2]["amplitude"], "m1_amplitude": modes[1]["amplitude"],
            "dominant_to_next_amplitude": ratio, "flags": " | ".join(flags)})
        diag = spiral["diagnostics"]
        pitch = diag["pitch_grid_deg"]
        for m, style in [(1, "--"), (2, "-")]:
            for mask in (pitch < 0, pitch > 0):
                ax.plot(pitch[mask], diag["spectra"][str(m)][mask], style,
                        color="tab:blue" if m == 2 else "tab:orange",
                        label=f"m={m}" if (pitch[mask] > 0).all() else None)
        ax.set(title=f"M51 {band}: same geometry + annulus", xlabel="Signed candidate pitch [deg]",
               ylabel="Relative Fourier amplitude")
        ax.legend()
    fig.savefig(OUT / "M51_common_geometry_crossband.png", dpi=150)
    plt.close(fig)
    write_csv(OUT / "M51_common_geometry_crossband.csv", rows)
    summary = {"common_geometry": common, "results": rows,
               "limitations": ["Geometry and annulus inherited from r band; no independent disk fit",
                   "No PSF matching, foreground masking or external pitch truth",
                   "Dominant-mode competition and changed band-specific annulus can change apparent pitch/mode",
                   "Spectra are exploratory descriptors, not distance predictions"]}
    save_json(OUT / "M51_common_geometry_crossband.json", summary)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_validation()
    manifest = ROOT / "executed_research" / "images" / "manifest.json"
    real = real_measurements(manifest)
    crossband = m51_crossband_comparison(manifest)
    readme = """# Calibrated-image morphology prototype

Run `OPENBLAS_NUM_THREADS=1 python experiments/morphology_experiment.py` from the repository root.
Dependencies: NumPy, SciPy, Astropy, Matplotlib. The script downloads nothing.

The `synthetic_validation.csv` records every attempted case, including failures and incorrect modes.
The supplied-geometry cases check pitch recovery under injected inclination, rotation, PSF blur,
noise and resampling. The automatic-moment cases deliberately test a weaker geometry estimator;
they must not be pooled with the supplied-geometry cases to advertise accuracy. This simulated
field lacks real dust, clumps, stars, bulges/bars and instrumental sky structure.

Real input manifest: `executed_research/images/manifest.json` with an `images` list, each entry:
`{"id":"galaxy_r","path":"executed_research/images/galaxy_r.fits","ra_deg":202.4,"dec_deg":47.2,"band":"r","source_url":"..."}`.
Optional: `hdu`, `center_xy`, `inclination_deg`, `pa_sky_deg` (north through east),
`pa_deg` (image +x toward +y), `pixel_scale_arcsec`, `r_inner_px`, `r_outer_px`,
`intrinsic_thickness` and `distance_mpc`. RA/Dec require WCS. Coordinates are zero-indexed.
The optional distance must be an external angular-diameter distance in Mpc, and is provenance,
not an image-model prediction. The WCS pixel scale assumes approximately square local pixels;
strong distortion/non-square pixels require reprojection before using this prototype.

No stretched JPEG is used for measurement. Border sky subtraction, central-aperture second
moments, finite aperture curve of growth and deprojected annular logarithmic Fourier modes
run on native FITS intensity pixels. No calibrated integrated flux or magnitude is claimed:
the implementation does not yet propagate photometric zero points or perform star deblending.
Moment inclination assumes a circular disk; a supplied intrinsic axial thickness is optional.
The moment PA and inclination can be biased by arms, bars, cutouts and foreground objects.

The strongest mode m and mean signed pitch satisfy tan(pitch)=-m/p. The measurement uses a
ring-normalized mean-subtracted intensity and radial taper, rather than exactly the raw flux
weighting of the original Fourier estimator. Pitch lies within a 5–60 degree search range.
Modes m=1..6 are explored, including lopsidedness m=1; mode m is not an arm-segmentation count.
The radial `periodic_azimuth_spacing_deg=360/m` and the deprojected periodic arc length are
descriptive periodicities, not individual arm boundaries or galaxy-pair physical distances.

The six sensitivity trials vary PA/inclination by +/-5 degrees and annulus endpoints by 15%.
Their min/max is a robustness diagnostic, not a confidence interval; centering, foreground,
background, PSF and model uncertainty remain. Low amplitude, edge-on geometry, weak sky,
bar competition, pitch-grid boundaries, missing pixels and unstable modes produce explicit flags.
Quality thresholds, including relative Fourier amplitude <0.03, are heuristics, not calibrated
detection significance. A numeric mode/pitch returned with a flag is a candidate descriptor.
Border sky may contain galaxy light. Reported r50/r90 use positive sky-subtracted pixels only
inside a conservative finite aperture and are not total-light half/effective radii.

Method reference: Savchenko & Reshetnikov (2013), MNRAS 436, 1074–1084,
https://academic.oup.com/mnras/article/436/2/1074/2907733 .

## Executed summary

```json
""" + json.dumps({"synthetic": synthetic, "real": real, "M51_crossband": crossband}, indent=2) + "\n```\n"
    (OUT / "README.txt").write_text(readme)
    print(json.dumps({"synthetic": synthetic, "real": real}, indent=2))


if __name__ == "__main__":
    main()
