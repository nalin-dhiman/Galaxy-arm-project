#!/usr/bin/env python3
"""Acquire/extract a fixed, label-blind morphology pilot from the SDSS sample.

The original first 160 rows are retained irrespective of download or quality
status. The selection does not inspect spec_z, magnitudes, physical sizes or
model predictions. Cutouts are fixed 256x256, 0.396 arcsec/pixel, Legacy DR9 r.
Usage: python experiments/sample_image_features.py acquire
       OPENBLAS_NUM_THREADS=1 python experiments/sample_image_features.py extract
No real-image arm/pitch ground truth is present. All quality gates are heuristic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlencode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.morphology import elliptical_radius, load_fits, measure_image, sky_center_and_pa

CATALOG = ROOT / "executed_research/images/sdss_resolved_sample.csv"
OUT = ROOT / "executed_research/sample_images"
PIXEL_SCALE = .396
SIDE = 256
COUNT = 160
ARM_KEYS = ["pitch_abs_deg", "arm_mode", "fourier_amplitude",
            "arm_radial_mode_support", "arm_mode_dominance_ratio"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_catalog(catalog=CATALOG, count=COUNT):
    with Path(catalog).open(newline="") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        rows = list(reader)
    if len(rows) < count:
        raise ValueError(f"Catalog has {len(rows)} rows; fixed pilot requires {count}")
    selected = []
    for index, row in enumerate(rows[:count]):
        # Explicit whitelist prevents labels entering acquisition or features.
        selected.append({"selection_row": index, "galaxy_id": row["galaxy_id"],
            "specObjID": row["specObjID"], "ra_deg": float(row["ra"]),
            "dec_deg": float(row["dec"]),
            "angular_r50_arcsec": float(row["angular_r50_arcsec"])})
    if len({row["galaxy_id"] for row in selected}) != count:
        raise ValueError("Duplicate galaxy IDs in fixed selection")
    return selected


def image_url(row):
    return "https://www.legacysurvey.org/viewer/fits-cutout?" + urlencode({
        "ra": f"{row['ra_deg']:.8f}", "dec": f"{row['dec_deg']:.8f}",
        "layer": "ls-dr9", "pixscale": PIXEL_SCALE, "size": SIDE, "bands": "r"})


def inspect_cached(path):
    image = load_fits(path)
    if image.data.shape != (SIDE, SIDE):
        raise ValueError(f"Unexpected cutout shape {image.data.shape}")
    if image.metadata.get("wcs") is None:
        raise ValueError("Cutout lacks celestial WCS")
    finite = image.data[np.isfinite(image.data)]
    if finite.size < .95 * SIDE * SIDE or np.ptp(finite) <= 0:
        raise ValueError("Blank/nonfinite image or insufficient coverage")
    return {"bytes": path.stat().st_size, "sha256": sha256(path),
            "finite_fraction": image.metadata["finite_fraction"],
            "pixel_scale_wcs_arcsec": image.metadata.get("pixel_scale_arcsec"),
            "bunit_header": image.metadata.get("bunit"), "hdu": image.metadata["hdu"]}


def acquire_one(row, out, offline=False):
    entry = {**row, "source_url": image_url(row), "band": "r",
             "survey_release": "Legacy Surveys DR9", "pixel_scale_arcsec": PIXEL_SCALE}
    path = out / "fits" / f"{row['galaxy_id']}_r.fits"
    entry["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    try:
        if path.exists():
            try:
                info = inspect_cached(path)
                return {**entry, **info, "acquisition_status": "cached"}
            except Exception as error:
                if offline:
                    raise ValueError(f"Invalid cached image: {error}")
        if offline:
            raise FileNotFoundError("Not cached; --offline disables requests")
        import requests
        started = time.monotonic()
        response = requests.get(entry["source_url"], timeout=(10, 30),
                                headers={"User-Agent": "galaxy-morphology-research-pilot/1.0"})
        response.raise_for_status()
        if len(response.content) > 5_000_000:
            raise ValueError("Unexpected image payload larger than 5 MB")
        temporary = path.with_suffix(".fits.part")
        temporary.write_bytes(response.content)
        info = inspect_cached(temporary)
        temporary.replace(path)
        return {**entry, **info, "acquisition_status": "downloaded",
                "elapsed_seconds": round(time.monotonic() - started, 3)}
    except Exception as error:
        return {**entry, "acquisition_status": "failed",
                "failure": f"{type(error).__name__}: {error}"}


def acquire(catalog, out, workers=3, offline=False):
    selected = select_catalog(catalog)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fits").mkdir(exist_ok=True)
    header = {"catalog": str(catalog), "catalog_sha256": sha256(catalog),
              "selection": "First 160 rows in supplied original checksum-based query ordering; no labels consulted; no replacements",
              "requested_rows": COUNT, "source_documentation": "https://www.legacysurvey.org/viewer/urls",
              "fixed_cutout": {"side_pixels": SIDE, "pixel_scale_arcsec": PIXEL_SCALE, "band": "r", "layer": "ls-dr9"},
              "timeouts_seconds": {"connect": 10, "read": 30}, "max_workers": min(workers, 3),
              "images": []}
    results = {}
    with ThreadPoolExecutor(max_workers=min(max(1, workers), 3)) as pool:
        pending = {pool.submit(acquire_one, row, out, offline): row["selection_row"] for row in selected}
        for future in as_completed(pending):
            index = pending[future]
            results[index] = future.result()
            if len(results) % 10 == 0 or len(results) == len(selected):
                header["images"] = [results[i] for i in sorted(results)]
                header["completed_requests"] = len(results)
                atomic_json(out / "manifest.json", header)
                good = sum(r["acquisition_status"] != "failed" for r in results.values())
                print(f"acquire {len(results)}/{len(selected)} finished; {good} valid cached/downloaded", flush=True)
    header["images"] = [results[i] for i in range(len(selected))]
    atomic_json(out / "manifest.json", header)
    return header


def extract_one(entry):
    row = {key: entry[key] for key in ("selection_row", "galaxy_id", "specObjID")}
    row.update({key: None for key in ARM_KEYS})
    row.update({"image_available": 0, "arm_candidate_accepted": 0,
                "extraction_status": "not_extracted", "quality_flags": ""})
    detail = {"input": entry}
    if entry.get("acquisition_status") == "failed":
        row.update(extraction_status="download_failed", quality_flags=entry.get("failure", "acquisition failed"))
        detail["features"] = row
        return row, detail
    try:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        image = load_fits(path, entry.get("hdu"))
        row["image_available"] = 1
        center, _ = sky_center_and_pa(image.metadata, entry["ra_deg"], entry["dec_deg"])
        edge = min(center[0], image.data.shape[1] - 1 - center[0],
                   center[1], image.data.shape[0] - 1 - center[1])
        catalog_r50_px = entry["angular_r50_arcsec"] / PIXEL_SCALE
        inner = max(6., 1.5 * catalog_r50_px)
        outer = min(.85 * edge, 3.5 * catalog_r50_px)
        row.update({"annulus_inner_arcsec": float(inner * PIXEL_SCALE),
                    "annulus_outer_arcsec": float(outer * PIXEL_SCALE)})
        if outer / inner < 1.5:
            raise ValueError("Angular annulus too narrow after fixed-image boundary clipping")
        result = measure_image(image.data, pixel_scale_arcsec=PIXEL_SCALE, center_xy=center,
            r_inner_px=inner, r_outer_px=outer, sensitivity=False)
        # Save detailed diagnostic candidates, but never expose rejected pitch,
        # mode or amplitude as accepted model inputs.
        spiral = result["spiral"]
        flags = list(result["flags"])
        ordered_modes = sorted(spiral["modes"], key=lambda m: m["amplitude"], reverse=True)
        dominance = ordered_modes[0]["amplitude"] / max(ordered_modes[1]["amplitude"], 1e-12)
        support = float(np.mean([item["mode_m"] == spiral["mode_m"] for item in spiral["radial_mode_spacing"]]))
        if support < 4 / 6:
            flags.append("RADIAL_DOMINANT_MODE_UNSTABLE")
        if dominance < 1.25:
            flags.append("COMPETING_FOURIER_MODES_WITHIN_25PCT")
        if spiral["mode_m"] == 1:
            flags.append("M1_LOPSIDEDNESS_NOT_MULTIPLE_ARMS")
        bad_flags = ("WEAK_PERIODIC_STRUCTURE", "BAR_OR_NONWINDING_STRUCTURE_COMPETES_WITH_SPIRAL",
            "PITCH_AT_SEARCH_BOUNDARY", "HIGH_INCLINATION_SENSITIVE_TO_DEPROJECTION",
            "LOW_ANNULUS_SURFACE_BRIGHTNESS", "RADIAL_DOMINANT_MODE_UNSTABLE",
            "COMPETING_FOURIER_MODES_WITHIN_25PCT", "M1_LOPSIDEDNESS_NOT_MULTIPLE_ARMS",
            "MISSING_ANNULUS_PIXELS_CAN_ALIAS_FOURIER_MODES")
        rejected = any(flag in flags for flag in bad_flags)
        radius_map = elliptical_radius(image.data.shape, center,
            result["pa_image_deg_used"], result["inclination_deg_used"])
        annulus_valid = np.isfinite(image.data) & (radius_map > inner) & (radius_map < outer)
        row.update(extraction_status="candidate_rejected" if rejected else "candidate_accepted",
                   arm_candidate_accepted=int(not rejected),
                   quality_flags=" | ".join(flags),
                   moment_axis_ratio=result["axis_ratio_moment"],
                   moment_inclination_deg=result["inclination_deg_used"],
                   image_annulus_snr_median=float(np.nanmedian((image.data - result["background"])[
                       annulus_valid])) / max(result["background_sigma"], 1e-12))
        if not rejected:
            row.update(pitch_abs_deg=spiral["pitch_abs_deg"], arm_mode=spiral["mode_m"],
                fourier_amplitude=spiral["amplitude"], arm_radial_mode_support=support,
                arm_mode_dominance_ratio=float(dominance))
        detail["measurement"] = {k: v for k, v in result.items() if k != "spiral"}
        detail["candidate_spiral"] = {k: v for k, v in spiral.items() if k != "diagnostics"}
        detail["quality_gates"] = {"radial_mode_support": support, "dominance_ratio": float(dominance),
                                   "rejected": rejected, "flags": flags}
    except Exception as error:
        row.update(extraction_status="measurement_failed", quality_flags=f"{type(error).__name__}: {error}")
    detail["features"] = row
    return row, detail


def extract(catalog, out):
    selected = select_catalog(catalog)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["catalog_sha256"] != sha256(catalog):
        raise ValueError("Catalog checksum differs from acquisition; refuse silently changed selection")
    entries = {entry["galaxy_id"]: entry for entry in manifest["images"]}
    rows, details = [], []
    for selected_row in selected:
        entry = entries.get(selected_row["galaxy_id"], {
            **selected_row, "acquisition_status": "failed", "failure": "Missing manifest acquisition result"})
        row, detail = extract_one(entry)
        rows.append(row)
        details.append(detail)
        if len(rows) % 20 == 0:
            print(f"extract {len(rows)}/{len(selected)} finished", flush=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "image_features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / "image_features.json", details)
    flag_counts = Counter(flag for row in rows for flag in row["quality_flags"].split(" | ") if flag)
    atomic_json(out / "quality_flag_counts.json", dict(flag_counts.most_common()))
    statuses = {status: sum(row["extraction_status"] == status for row in rows)
                for status in sorted({row["extraction_status"] for row in rows})}
    summary = {"catalog_sha256": sha256(catalog), "rows_preserved": len(rows),
        "images_available": sum(row["image_available"] for row in rows),
        "accepted_arm_candidates": sum(row["arm_candidate_accepted"] for row in rows),
        "statuses": statuses, "arm_feature_columns": ARM_KEYS,
        "limitations": ["No independent arm tracing, pitch labels, PSF, inverse variance or survey masks",
            "Fixed observed angular cutouts; annulus uses catalog angular R50, never physical kpc or redshift",
            "Native image intensities; no validated zero-point photometry or uncertainty propagation",
            "Automatic moment inclination assumes a circular disk and can be biased by bars, stars and bulges",
            "Quality gates are heuristic, not detection confidence",
            "Missing arm features require train-only imputation and identical held-out galaxies for baseline comparisons",
            "Download/acceptance can be redshift-dependent; compare missingness and no-survey-overlap sensitivity",
            "Exploratory selected sample, not external generalization or distance-accuracy validation"]}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    plot_quality_examples(details, out)
    return summary


def plot_quality_examples(details, out):
    """Display first available accepted/rejected examples, with no relabeling."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    accepted = [d for d in details if d.get("features", {}).get("arm_candidate_accepted") == 1]
    rejected = [d for d in details if d.get("features", {}).get("image_available") == 1
                and d.get("features", {}).get("arm_candidate_accepted") == 0]
    chosen = (accepted[:4] + rejected[:max(0, 8 - len(accepted[:4]))])[:8]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, detail in zip(axes.ravel(), chosen):
        entry, row = detail["input"], detail["features"]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        image = load_fits(path).data
        center = ((SIDE - 1) / 2, (SIDE - 1) / 2)
        measured = detail.get("measurement")
        if measured:
            center = (measured["center_x_px"], measured["center_y_px"])
            background = measured["background"]
        else:
            background = float(np.nanmedian(image))
        half = min(90, max(30, 5 * entry["angular_r50_arcsec"] / PIXEL_SCALE))
        yy, xx = np.indices(image.shape)
        crop = (np.abs(xx - center[0]) < half) & (np.abs(yy - center[1]) < half)
        values = (image - background)[crop & np.isfinite(image)]
        scale = max(float(np.percentile(values, 95)) / 4, 1e-10)
        ax.imshow(np.arcsinh((image - background) / scale), origin="lower", cmap="gray", vmin=-.2,
                  vmax=np.arcsinh(max(float(np.percentile(values, 99)), scale) / scale))
        if measured:
            pa, inc = np.radians(measured["pa_image_deg_used"]), np.radians(measured["inclination_deg_used"])
            theta = np.linspace(0, 2 * np.pi, 200)
            for radius in (measured["r_inner_px"], measured["r_outer_px"]):
                X, Y = radius * np.cos(theta), radius * np.sin(theta) * np.cos(inc)
                ax.plot(center[0] + np.cos(pa) * X - np.sin(pa) * Y,
                        center[1] + np.sin(pa) * X + np.cos(pa) * Y, color="tomato", lw=.7)
        ax.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half))
        status = "Accepted candidate" if row["arm_candidate_accepted"] else "Rejected/failed"
        title = f"{status}; source row {entry['selection_row']}"
        if row["arm_candidate_accepted"]:
            title += f"\nm={row['arm_mode']}, pitch={row['pitch_abs_deg']:.1f}°"
        ax.set_title(title, fontsize=10)
    fig.suptitle("Fixed sample: first accepted/rejected cases in selection order; display stretch only\nNo independent arm labels; red ellipses show automatic measurement annulus", fontsize=12)
    fig.savefig(out / "quality_examples.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "extract"))
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.catalog, args.out, args.workers, args.offline)
    else:
        extract(args.catalog, args.out)


if __name__ == "__main__":
    main()
