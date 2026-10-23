#!/usr/bin/env python3
"""Execute controlled positives/negatives; fixed development/test protocol.

These truth masks come from the injection model, not independent observations.
They validate code behavior and expose false positives; observational evidence
must come from independent human spiral traces. No image-to-distance claim.
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
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.filtering import (ScienceImage, DiskGeometry, filter_bank,
    evaluate_score, foreground_point_mask, radius_map, display_transform,
    refine_outer_geometry)
OUT = ROOT / "executed_filters" / "synthetic"
SCALES = (.4, .8, 1.6)  # sky sigmas, fixed before generation/evaluation
PIXEL_SCALE = .3


def make_case(kind, index, band="r"):
    shape = (192, 192); yy, xx = np.indices(shape); center = (95.5, 95.5)
    pa = [20, 65, 110, 150][index % 4]
    inc = [20, 35, 50, 60][index % 4]
    angle = np.radians(pa)
    xp = (xx - center[0]) * np.cos(angle) + (yy - center[1]) * np.sin(angle)
    yp = (-(xx - center[0]) * np.sin(angle) + (yy - center[1]) * np.cos(angle)) / np.cos(np.radians(inc))
    radius = np.hypot(xp, yp); theta = np.arctan2(yp, xp)
    pitch = [12, 20, 28, 36][index % 4]
    mode = 2 if index % 3 else 3
    phase = mode * (theta - np.log(np.maximum(radius, 1) / 35) / np.tan(np.radians(pitch)))
    armshape = (np.exp(4 * np.cos(phase)) - np.exp(-4)) / (np.exp(4) - np.exp(-4))
    taper = (1 / (1 + np.exp(-(radius - 25) / 3))) * (1 / (1 + np.exp((radius - 85) / 3)))
    profile = np.exp(-radius / 45)
    signal = 1.5 * profile + 2 * np.exp(-.5 * (radius / 8) ** 2)
    truth = np.zeros(shape, bool)
    if kind == "spiral":
        contrast = {"g": 1.4, "r": 1., "i": .65}[band]
        signal += contrast * profile * armshape * taper
        truth = (armshape > .3) & (radius > 25) & (radius < 85)
    elif kind == "bar":
        signal += .7 * np.exp(-.5 * (xp / 45) ** 8 - .5 * (yp / 4) ** 2)
    elif kind == "ring":
        signal += .65 * np.exp(-.5 * ((radius - 58) / 3) ** 2)
    elif kind == "foreground_star":
        signal += 5 * np.exp(-.5 * ((xx - 141) ** 2 + (yy - 68) ** 2) / .35 ** 2)
    elif kind != "smooth":
        raise ValueError(kind)
    fwhm = {"g": 1.5, "r": 1.2, "i": 1.0}[band]
    signal = gaussian_filter(signal, fwhm / 2.354820045 / PIXEL_SCALE)
    noise = {"g": .045, "r": .03, "i": .025}[band]
    data = signal + .13 + np.random.default_rng(441 + index).normal(0, noise, shape)
    image = ScienceImage(data, PIXEL_SCALE, unit="synthetic linear intensity", band=band,
        variance=np.full(shape, noise ** 2), psf_fwhm_arcsec=fwhm,
        metadata={"injection_kind": kind})
    geometry = DiskGeometry(center, pa, inc, 8, 24, "known injected geometry")
    return image, truth, geometry, {"pitch_deg": pitch if kind == "spiral" else None,
        "inclination_deg": inc, "pa_deg": pa, "PSF_FWHM_arcsec": fwhm,
        "injected_noise": noise, "model_arm_truth": "intrinsic arm-component level >0.3, independent of filtering"}


def normalized(score, valid):
    finite = score[valid & np.isfinite(score)]
    med = np.median(finite)
    mad = max(1.4826 * np.median(abs(finite - med)), 1e-12)
    return (score - med) / mad


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"filter_scales_sigma_arcsec": SCALES, "pixel_scale_arcsec": PIXEL_SCALE,
        "positives": "8 injected spirals x g/r/i bands; indices 0:3 development,4:7 held-out",
        "negative_controls": "smooth disks, bars, rings, and foreground stars, each 4 geometries; index 0:1 development,2:3 held-out",
        "selection": "single filter maximizing mean development AP over positive galaxies; threshold maximizing development F1 pooled over positive and control images, after within-image annular median/MAD normalization",
        "parameter_frozen_before_test": True,
        "limitations": ["Analytic injections are not independent observational validation.",
            "Band images share injected structures; they are correlated observations, not independent galaxies.",
            "A ring or bar response is a false spiral candidate; filters are not arm classifiers.",
            "Pixel correlations require galaxy-level observational uncertainty analysis."]}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    cases = []
    for index in range(8):
        for band in ("g", "r", "i"):
            cases.append(("spiral", index, band, "development" if index < 4 else "test"))
    for kind in ("smooth", "bar", "ring", "foreground_star"):
        for index in range(4):
            cases.append((kind, index, "r", "development" if index < 2 else "test"))
    rows, cache = [], {}
    for kind, index, band, split in cases:
        image, truth, geometry, extra = make_case(kind, index, band)
        maps = filter_bank(image, geometry, SCALES)
        key = f"{kind}_{index:02d}_{band}"
        for name, result in maps.items():
            metric = evaluate_score(result.score, truth, result.valid)
            rows.append({"id": key, "kind": kind, "band": band, "split": split,
                         "filter": name, **extra, **metric})
        cache[key] = (image, truth, geometry, maps, kind, split, band)
    # This is the only model selection, and consults development truth only.
    names = list(next(iter(cache.values()))[3])
    development = {name: float(np.mean([r["average_precision"] for r in rows
        if r["filter"] == name and r["split"] == "development" and r["kind"] == "spiral"])) for name in names}
    selected = max(development, key=development.get)
    dev_scores, dev_truth = [], []
    for image, truth, geo, maps, kind, split, band in cache.values():
        if split != "development":
            continue
        item = maps[selected]; z = normalized(item.score, item.valid)
        dev_scores.append(z[item.valid]); dev_truth.append(truth[item.valid])
    precision, recall, thresholds = precision_recall_curve(np.concatenate(dev_truth), np.concatenate(dev_scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[int(np.argmax(f1))])
    selected_rows = []
    for key, (image, truth, geo, maps, kind, split, band) in cache.items():
        item = maps[selected]; z = normalized(item.score, item.valid)
        metrics = evaluate_score(z, truth, item.valid, threshold=threshold)
        selected_rows.append({"id": key, "kind": kind, "band": band, "split": split,
            "selected_filter": selected, **metrics,
            "fraction_pixels_called_candidate": float(np.mean(z[item.valid] >= threshold))})
    save_csv(OUT / "all_filters.csv", rows)
    save_csv(OUT / "selected_filter_predictions.csv", selected_rows)
    summaries = {}
    for band in ("g", "r", "i"):
        sub = [r for r in selected_rows if r["split"] == "test" and r["kind"] == "spiral" and r["band"] == band]
        summaries[band] = {"test_galaxies": len(sub), "mean_AP": float(np.mean([r["average_precision"] for r in sub])),
            "mean_precision": float(np.mean([r["precision"] for r in sub])), "mean_recall": float(np.mean([r["recall"] for r in sub]))}
    control_summary = {}
    for kind in ("smooth", "bar", "ring", "foreground_star"):
        sub = [r for r in selected_rows if r["split"] == "test" and r["kind"] == kind]
        control_summary[kind] = {"test_images": len(sub), "mean_false_candidate_fraction": float(np.mean([r["fraction_pixels_called_candidate"] for r in sub]))}
    # A direct foreground-star check, with explicit unmasked/masked presentation.
    image, truth, geo, _ = make_case("foreground_star", 2)
    mask, stars = foreground_point_mask(image, geo.center_xy, psf_fwhm_arcsec=image.psf_fwhm_arcsec)
    star_summary = {"known_injected_star_x": 141, "known_injected_star_y": 68,
                    "star_position_masked": bool(mask[68, 141]), "detections": stars,
                    "limitation": "PSF-like knots may be confused with stars; independent catalogs/masks remain necessary"}
    # Check outer-light ellipse refinement on a positive, independent of arm labels.
    image, truth, geo, _ = make_case("spiral", 2)
    initial = DiskGeometry(geo.center_xy, geo.pa_deg - 8, geo.inclination_deg - 6,
                           geo.r_inner_arcsec, geo.r_outer_arcsec, "perturbed injection geometry")
    refined, diag = refine_outer_geometry(image, initial)
    geometry_summary = {"true_PA": geo.pa_deg, "initial_PA": initial.pa_deg, "refined_PA": refined.pa_deg,
        "true_inclination": geo.inclination_deg, "initial_inclination": initial.inclination_deg,
        "refined_inclination": refined.inclination_deg, **diag}
    summary = {"selected_filter": selected, "development_mean_AP": development,
        "threshold_z_annular_median_MAD": threshold,
        "heldout_positive_results_by_band": summaries,
        "heldout_negative_controls": control_summary,
        "foreground_mask_test": star_summary, "outer_isophote_sensitivity_example": geometry_summary,
        "interpretation": "Algorithm validation only. Filters highlight non-spiral rings/bars/foreground sources too; no distance or fluid inference follows."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Display single positive and representative nulls with the single frozen map.
    examples = ["spiral_04_r", "smooth_02_r", "bar_02_r", "ring_02_r", "foreground_star_02_r"]
    fig, axes = plt.subplots(len(examples), 4, figsize=(13, 14), constrained_layout=True)
    for row, key in enumerate(examples):
        image, truth, geo, maps, kind, split, band = cache[key]
        item = maps[selected]; z = normalized(item.score, item.valid)
        axes[row, 0].imshow(display_transform(image.data), origin="lower", cmap="gray")
        axes[row, 0].set_title(key.replace("_", " "))
        axes[row, 1].imshow(truth & item.valid, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Known injected arm mask")
        axes[row, 2].imshow(z, origin="lower", cmap="magma", vmin=-1, vmax=5)
        axes[row, 2].set_title(f"Frozen {selected} score")
        axes[row, 3].imshow((z >= threshold) & item.valid, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"Frozen threshold z={threshold:.2f}")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Injection diagnostics: structure response does not identify a spiral", fontsize=14)
    fig.savefig(OUT / "positive_negative_diagnostics.png", dpi=140); plt.close(fig)
    # All filter scores, for inspectability (no hand-picked per-test winner).
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    image, truth, geo, maps, *_ = cache["spiral_04_r"]
    for ax, (name, item) in zip(axes.flat, maps.items()):
        ax.imshow(normalized(item.score, item.valid), origin="lower", cmap="magma", vmin=-1, vmax=5)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Fixed angular-scale filter bank on held-out synthetic spiral")
    fig.savefig(OUT / "filter_bank.png", dpi=140); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
