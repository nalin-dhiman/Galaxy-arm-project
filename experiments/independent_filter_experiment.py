#!/usr/bin/env python3
"""Validate fixed filters against independent GZ3D human arm polygons.

Native science images stay on the r-band survey pixel grid. Only human labels
and rendered RGB display pixels are mapped to that grid. No physical distance,
arm pitch truth, or viscosity is inferred by this experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.filtering import (ScienceImage, DiskGeometry, filter_bank,
    display_grayscale, display_transform, evaluate_score, match_gaussian_psf,
    inverse_variance_coadd, radius_map)

OUT = ROOT / "executed_filters" / "benchmark"
SCALES = (.4, .8, 1.6)
SEED = 20261023


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def normalize_score(score, valid):
    values = np.asarray(score)[valid & np.isfinite(score)]
    if len(values) < 20:
        raise ValueError("Insufficient valid score pixels")
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    # Sparse nonnegative ridge/edge maps can have zero MAD. This fallback is
    # image-only and fixed before evaluation; no label enters normalization.
    scale = max(mad, float(np.std(values)) * .1, 1e-12)
    return (score - median) / scale, {"score_median": median, "score_scale": scale}


def label_coordinates(source_wcs, target_wcs, shape):
    yy, xx = np.indices(shape)
    sky = target_wcs.pixel_to_world(xx, yy)
    xs, ys = source_wcs.world_to_pixel(sky)
    return xs, ys


def map_labels(values, xs, ys):
    """Nearest human vote counts, with explicit coverage; no label smoothing."""
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (ys >= 0)
    valid &= (xs <= values.shape[1] - 1) & (ys <= values.shape[0] - 1)
    coords = [np.where(valid, ys, 0), np.where(valid, xs, 0)]
    mapped = map_coordinates(np.asarray(values, float), coords, order=0,
                             mode="constant", cval=0, prefilter=False)
    return mapped, valid


def parse_native(path, record):
    """Read unresampled coadd image/invvar HDUs; reject absent metadata."""
    images = {}
    psfs = record.get("psf_fwhm_arcsec", {})
    if isinstance(psfs, (int, float)):
        psfs = {b: float(psfs) for b in "grz"}
    band_records = record.get("bands", [])
    if isinstance(band_records, dict):
        band_records = [{"name": k, **v} for k, v in band_records.items()]
    for b in band_records:
        if isinstance(b, dict) and b.get("psf_fwhm_arcsec") is not None:
            psfs[b.get("name", b.get("band"))] = b["psf_fwhm_arcsec"]
    with fits.open(path) as hdus:
        planes = {}
        for h in hdus:
            band = str(h.header.get("BAND", "")).strip()
            kind = str(h.header.get("IMAGETYP", "")).strip()
            if band in "grz" and band and h.data is not None and h.data.ndim == 2:
                planes[band, kind] = (np.array(h.data, float), h.header.copy())
        for band in "grz":
            data, header = planes[band, "image"]
            invvar, _ = planes[band, "invvar"]
            wcs = WCS(header).celestial
            scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600)
            good = np.isfinite(data) & np.isfinite(invvar) & (invvar > 0)
            variance = np.divide(1., invvar, out=np.full(data.shape, np.nan), where=good)
            images[band] = ScienceImage(data, scale, unit="nanomaggy/pixel",
                mask=~good, variance=variance, wcs=wcs, band=band,
                psf_fwhm_arcsec=psfs.get(band), pixel_value_kind="flux_per_pixel",
                metadata={"unit_source": record.get("unit_source", "Legacy Surveys DR9 image documentation"),
                          "native_file": str(path), "no_science_reprojection": True})
    ref = images["r"]
    # A full 3x3 grid detects rotations/shears as well as translations.
    yy, xx = np.meshgrid(np.linspace(0, ref.data.shape[0] - 1, 3),
                         np.linspace(0, ref.data.shape[1] - 1, 3), indexing="ij")
    for band, im in images.items():
        if im.data.shape != ref.data.shape:
            raise ValueError(f"{band}: native band shapes differ; science reprojection prohibited here")
        offset = ref.wcs.pixel_to_world(xx, yy).separation(im.wcs.pixel_to_world(xx, yy)).arcsec
        if np.max(offset) > .001 * ref.pixel_scale_arcsec:
            raise ValueError(f"{band}: native grids differ; do not invent common-grid photometry")
    return images


def load_object(annotation, native, *, psf_match=True):
    images = parse_native(ROOT / native["path"], native)
    reference = images["r"]
    with fits.open(ROOT / annotation["path"]) as hdus:
        awcs = WCS(hdus[annotation["wcs_hdu"]].header).celestial
        xs, ys = label_coordinates(awcs, reference.wcs, reference.data.shape)
        votes, label_valid = map_labels(hdus[annotation["spiral_hdu"]].data, xs, ys)
        stars, _ = map_labels(hdus[annotation["stars_hdu"]].data, xs, ys)
        rgb = np.asarray(hdus[annotation["rgb_hdu"]].data, float)
        rgb_native = np.stack([map_coordinates(rgb[..., k], [ys, xs], order=1,
            mode="constant", cval=np.nan, prefilter=False) for k in range(3)], axis=-1)
        # Fixed color rule identifies the burned-in magenta telescope outline.
        # This is display-artifact removal, never a native science-pixel mask.
        red, green, blue = rgb_native.transpose(2, 0, 1)
        artifact = (red > 70) & (blue > 70) & (np.minimum(red, blue) > 1.4 * green + 15)
        center_sky = awcs.pixel_to_world(annotation["center_x"], annotation["center_y"])
        center = tuple(float(x) for x in reference.wcs.world_to_pixel(center_sky))
    for image in images.values():
        image.mask = ~image.valid | (stars > 0)
    input_psfs = {b: im.psf_fwhm_arcsec for b, im in images.items()}
    if psf_match:
        if any(x is None for x in input_psfs.values()):
            raise ValueError("Missing measured PSF; no PSF-matched comparison possible")
        target_psf = float(max(input_psfs.values()))
        images = {b: match_gaussian_psf(im, target_psf) for b, im in images.items()}
        images["grz_detection"] = inverse_variance_coadd(list(images.values()))
    geometry = DiskGeometry(center, 0., 0., 3., 23., "human center; fixed circular sky annulus")
    science_maps = {}
    for band, image in images.items():
        science_maps.update({band + "/" + key: val for key, val in filter_bank(image, geometry, SCALES).items()})
    common = label_valid.copy()
    for item in science_maps.values():
        common &= item.valid
    for item in science_maps.values():
        item.valid &= common
        item.score[~common] = np.nan
    if common.sum() < 100:
        raise ValueError("Fewer than 100 common native pixels survive fixed coverage guards")
    raw_r = images["r"]
    # Contrast transforms operate on independent display copies. The global
    # asinh stretch approximately preserves ranks; CLAHE need not preserve them.
    for method in ("linear", "asinh", "clahe"):
        display = display_transform(raw_r.data, method)
        from galaxy_lab.filtering import FilterMap
        science_maps["r_display/" + method] = FilterMap(np.where(common, display, np.nan), common.copy(),
            {"provenance": "display transform of r science image; not calibrated flux"})
    gray = ScienceImage(display_grayscale(rgb_native), reference.pixel_scale_arcsec,
        mask=~label_valid | artifact | (stars > 0), provenance="rendered_display",
        band="GZ3D_rendered_gray", pixel_value_kind="surface_brightness")
    # Rendered comparison is secondary: its artifact mask has separate coverage.
    gray_failure = None
    try:
        gray_maps = filter_bank(gray, geometry, SCALES)
    except ValueError as exc:
        gray_maps = {}
        gray_failure = str(exc)
    geometric = radius_map(reference.data.shape, geometry, reference.pixel_scale_arcsec)
    geometric = (geometric >= 3) & (geometric <= 23) & label_valid
    details = {"id": annotation["id"], "sample_group": annotation["sample_group"],
        "split": annotation["split"], "input_psf_fwhm_arcsec": input_psfs,
        "psf_match": psf_match, "psf_model": "circular Gaussian approximation to measured coadd PSF moments",
        "pixel_scale_arcsec": reference.pixel_scale_arcsec,
        "center_x_native": center[0], "center_y_native": center[1],
        "geometric_annulus_pixels": int(geometric.sum()), "common_science_pixels": int(common.sum()),
        "science_coverage": float(common.sum() / geometric.sum()) if geometric.any() else None,
        "human_positive_ge3_before_guard": int(((votes >= 3) & geometric).sum()),
        "human_positive_ge3_after_guard": int(((votes >= 3) & common).sum()),
        "rendered_magenta_pixels": int((artifact & geometric).sum()),
        "rendered_valid_pixels": int((gray_maps["native_intensity"].valid & common).sum()) if gray_maps else 0,
        "rendered_failure": gray_failure,
        "spatial_label_status": annotation.get("spatial_label_status"),
        "science_mask": "native invalid + independent human star votes, 4-largest-sigma guard",
        "rendered_mask": "independent stars + fixed magenta-overlay color rule, same guard"}
    return {"annotation": annotation, "images": images, "votes": votes, "common": common,
            "maps": science_maps, "gray_maps": gray_maps, "gray": gray, "rgb": rgb_native,
            "geometry": geometry, "details": details}


def score_object(obj):
    a = obj["annotation"]
    rows = []
    for pipeline, item in obj["maps"].items():
        for label_threshold in (3, 5):
            result = evaluate_score(item.score, obj["votes"] >= label_threshold, item.valid)
            if a["sample_group"] != "strong_spiral":
                for key in result:
                    if key != "valid_pixels":
                        result[key] = None
            ap, prevalence = result["average_precision"], result["positive_fraction"]
            rows.append({"id": a["id"], "sample_group": a["sample_group"], "split": a["split"],
                "pipeline": pipeline, "label_votes": label_threshold, **result,
                "AP_over_prevalence": ap / prevalence if ap is not None and prevalence else None})
    return rows


def select_development(rows):
    """Require the same complete development galaxies for every contender."""
    df = pd.DataFrame(rows)
    df = df[(df.split == "development") & (df.sample_group == "strong_spiral") & (df.label_votes == 3)]
    pivot = df.pivot(index="id", columns="pipeline", values="average_precision")
    if pivot.empty or pivot.isna().any().any():
        raise ValueError("Incomplete development score matrix; cannot select across unequal cohorts")
    means = pivot.mean()
    selected = sorted(means.index, key=lambda key: (-means[key], key))[0]
    return selected, {str(k): float(v) for k, v in means.items()}, list(pivot.index)


def binary_threshold(objects, selected):
    scores, truths = [], []
    for obj in objects:
        a = obj["annotation"]
        if a["split"] != "development" or a["sample_group"] != "strong_spiral":
            continue
        item = obj["maps"][selected]
        z, _ = normalize_score(item.score, item.valid)
        scores.append(z[item.valid]); truths.append((obj["votes"] >= 3)[item.valid])
    precision, recall, thresholds = precision_recall_curve(np.concatenate(truths), np.concatenate(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(f1))])


def paired_interval(differences):
    x = np.asarray(differences, float)
    if len(x) < 2:
        return {"galaxies": len(x), "mean_difference": float(np.mean(x)) if len(x) else None,
                "bootstrap_95_interval": None}
    rng = np.random.default_rng(SEED)
    draws = np.mean(rng.choice(x, size=(10000, len(x)), replace=True), axis=1)
    return {"galaxies": len(x), "mean_difference": float(np.mean(x)),
            "bootstrap_95_interval": [float(v) for v in np.quantile(draws, [.025, .975])]}


def summarize(all_rows, objects, frozen):
    df = pd.DataFrame(all_rows)
    test = df[(df.split == "test") & (df.sample_group == "strong_spiral") & (df.label_votes == 3)]
    pivot = test.pivot(index="id", columns="pipeline", values="average_precision")
    if pivot.isna().any().any():
        raise ValueError("Unequal test cohorts: report failures before making comparisons")
    selected = frozen["selected_pipeline"]
    rbase = "r/native_intensity"
    binary_rows, paired_gray = [], []
    for obj in objects:
        a = obj["annotation"]
        item = obj["maps"][selected]
        z, norm = normalize_score(item.score, item.valid)
        pred = z >= frozen["binary_threshold_z"]
        truth = obj["votes"] >= 3
        metric = evaluate_score(z, truth, item.valid, threshold=frozen["binary_threshold_z"])
        if a["sample_group"] != "strong_spiral":
            # No task participation means no spatial truth, not an arm-free map.
            for key in ("precision", "recall", "true_positive", "false_positive", "false_negative"):
                metric.pop(key, None)
            for key in ("positive_pixels", "positive_fraction", "average_precision", "roc_auc"):
                metric[key] = None
        valid = item.valid
        union = int(((pred | truth) & valid).sum())
        intersection = int(((pred & truth) & valid).sum())
        binary_rows.append({"id": a["id"], "split": a["split"], "sample_group": a["sample_group"],
            "spatial_label_status": a.get("spatial_label_status"), "selected_pipeline": selected,
            **metric, **norm, "candidate_pixel_fraction": float(np.mean(pred[valid])) if valid.any() else None,
            "IoU": intersection / union if union and a["sample_group"] == "strong_spiral" else None})
        # Both APs in each paired comparison use exactly the same pixels.
        if a["split"] == "test" and a["sample_group"] == "strong_spiral":
            if not obj["gray_maps"]:
                paired_gray.append({"id": a["id"], "status": "rendered_pipeline_failure",
                                    "reason": obj["details"]["rendered_failure"], "pixels": 0})
                continue
            gray = obj["gray_maps"]["native_intensity"]
            shared = valid & gray.valid
            if shared.sum() < 100:
                paired_gray.append({"id": a["id"], "status": "insufficient_shared_pixels", "pixels": int(shared.sum())})
                continue
            gm = evaluate_score(gray.score, truth, shared)
            sm = evaluate_score(item.score, truth, shared)
            if gm["average_precision"] is None or sm["average_precision"] is None:
                paired_gray.append({"id": a["id"], "status": "no_both_classes_on_shared_region", "pixels": int(shared.sum())})
                continue
            paired_gray.append({"id": a["id"], "status": "evaluated", "pixels": int(shared.sum()),
                "selected_AP": sm["average_precision"], "rendered_gray_AP": gm["average_precision"],
                "difference": sm["average_precision"] - gm["average_precision"]})
    save_csv(OUT / "binary_masks.csv", binary_rows)
    save_csv(OUT / "rendered_gray_paired.csv", paired_gray)
    test_binary = [r for r in binary_rows if r["split"] == "test" and r["sample_group"] == "strong_spiral"]
    controls = [r for r in binary_rows if r["split"] == "test" and r["sample_group"] != "strong_spiral"]
    sensitivity = {}
    for threshold in (3, 5):
        sub = df[(df.split == "test") & (df.sample_group == "strong_spiral") & (df.label_votes == threshold)]
        sens = sub.pivot(index="id", columns="pipeline", values="average_precision")
        differences = (sens[selected] - sens[rbase]).dropna()
        sensitivity[str(threshold)] = {"selected_defined_galaxies": int(sens[selected].notna().sum()),
            "selected_macro_AP": float(sens[selected].mean()), "r_macro_AP": float(sens[rbase].mean()),
            "paired_difference": paired_interval(differences)}
    return {"frozen_choice": frozen, "test_spirals_scored": len(pivot),
        "test_macro_AP_by_pipeline": {k: float(v) for k, v in pivot.mean().items()},
        "test_macro_prevalence": float(test[test.pipeline == selected].positive_fraction.mean()),
        "selected_vs_raw_r": paired_interval(pivot[selected] - pivot[rbase]),
        "selected_vs_rendered_gray_shared_region": paired_interval([r["difference"] for r in paired_gray if r["status"] == "evaluated"]),
        "rendered_comparison_failures": [r for r in paired_gray if r["status"] != "evaluated"],
        "label_threshold_sensitivity": sensitivity,
        "test_binary_macro": {k: float(np.mean([r[k] for r in test_binary])) for k in ("precision", "recall", "IoU", "candidate_pixel_fraction")},
        "test_catalog_controls": {"galaxies": len(controls), "mean_candidate_pixel_fraction": float(np.mean([r["candidate_pixel_fraction"] for r in controls])) if controls else None,
            "interpretation": "Low catalog spiral-vote objects; no spatial arm-task participation, so these are not confirmed negative masks or a false-positive-rate estimate."},
        "uncertainty": "Paired bootstrap of whole galaxies, 10,000 draws, fixed seed. Small selected sample; no pixel-independent confidence interval.",
        "interpretation": "Agreement with incomplete human arm regions conditional on supplied centers, stars, survey astrometry and common coverage. Not distance, pitch accuracy, viscosity, or survey-wide arm classification."}


def plots(objects, frozen, rows):
    test = [obj for obj in objects if obj["annotation"]["split"] == "test" and obj["annotation"]["sample_group"] == "strong_spiral"]
    selected = frozen["selected_pipeline"]
    fig, axes = plt.subplots(min(6, len(test)), 5, figsize=(14, 15), squeeze=False, constrained_layout=True)
    for i, obj in enumerate(test[:6]):
        a = obj["annotation"]; item = obj["maps"][selected]
        z, _ = normalize_score(item.score, item.valid)
        displays = [display_transform(obj["images"]["r"].data), obj["votes"],
                    item.valid, z, np.where(item.valid, z >= frozen["binary_threshold_z"], np.nan)]
        titles = [a["id"] + "  r asinh", "Human arm votes", "Evaluated region", selected, "Frozen binary candidates"]
        for j, data in enumerate(displays):
            axes[i, j].imshow(data, origin="lower", cmap="magma" if j in (1, 3) else "gray",
                              **({"vmin": -1, "vmax": 5} if j == 3 else {}))
            if j == 0:
                axes[i, j].contour(obj["votes"] >= 3, levels=[.5], colors="cyan", linewidths=.5)
            axes[i, j].set_title(titles[j], fontsize=9); axes[i, j].set_axis_off()
    fig.savefig(OUT / "heldout_gallery.png", dpi=155); plt.close(fig)
    # Deterministically first test galaxy, never chosen for best performance.
    obj = test[0]; r = obj["images"]["r"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for ax, method in zip(axes[0], ("linear", "asinh", "clahe", "binary")):
        ax.imshow(display_transform(r.data, method), origin="lower", cmap="gray")
        ax.set_title("r display: " + method); ax.set_axis_off()
    for ax, band in zip(axes[1], ("g", "r", "z", "grz_detection")):
        if band in obj["images"]:
            ax.imshow(display_transform(obj["images"][band].data), origin="lower", cmap="gray")
        ax.set_title(band + " (independent stretch)"); ax.set_axis_off()
    fig.suptitle(obj["annotation"]["id"] + ": display brightness is not cross-band calibrated photometry")
    fig.savefig(OUT / "passbands_grayscale_binary.png", dpi=155); plt.close(fig)
    df = pd.DataFrame(rows)
    good = df[(df.sample_group == "strong_spiral") & (df.label_votes == 3)]
    pivot = good.groupby(["pipeline", "split"]).average_precision.mean().unstack()
    order = pivot.development.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    y = np.arange(len(order))
    ax.plot(pivot.loc[order, "development"], y, "o", label="Development (selection)")
    ax.plot(pivot.loc[order, "test"], y, "s", label="Held-out galaxies")
    ax.set_yticks(y, order, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Macro average precision against >=3 human votes"); ax.legend()
    ax.set_title("All fixed pipelines; one shared development choice, no per-test winner")
    fig.savefig(OUT / "filter_comparison.png", dpi=155); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmatched-psf", action="store_true", help="Explicit sensitivity run; do not infer wavelength causes from unequal seeing")
    args = parser.parse_args()
    global OUT
    if args.unmatched_psf:
        OUT = ROOT / "executed_filters" / "benchmark_unmatched"
    OUT.mkdir(parents=True, exist_ok=True)
    annotations_path = ROOT / "executed_filters/annotations/manifest.json"
    native_path = ROOT / "executed_filters/native_images/manifest.json"
    annotation_manifest = json.loads(annotations_path.read_text())
    native_manifest = json.loads(native_path.read_text())
    native_records = native_manifest.get("images", native_manifest.get("targets", []))
    native = {r["id"]: r for r in native_records}
    annotations = annotation_manifest["images"]
    objects, rows, failures = [], [], []
    # Process development only, then save a concrete immutable choice before
    # any held-out filter score is calculated.
    for split in ("development", "test"):
        if split == "test":
            dev_spirals = [o for o in objects if o["annotation"]["split"] == "development" and o["annotation"]["sample_group"] == "strong_spiral"]
            if len(dev_spirals) != 12:
                (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                raise ValueError("The prespecified 12-galaxy development cohort is incomplete; inspect recorded failures")
            selected, means, ids = select_development(rows)
            threshold = binary_threshold(objects, selected)
            frozen = {"selected_pipeline": selected, "development_macro_AP_by_pipeline": means,
                "development_ids": ids, "binary_threshold_z": threshold,
                "binary_selection": "pooled development-positive F1 after label-free per-image median/MAD normalization; controls have no spatial truth",
                "scales_sigma_arcsec": SCALES, "annulus_arcsec": [3, 23],
                "label_votes": 3, "psf_match": not args.unmatched_psf,
                "annotation_manifest_sha256": sha(annotations_path), "native_manifest_sha256": sha(native_path),
                "heldout_scores_read_during_selection": False,
                "selection_code_sha256": sha(__file__)}
            (OUT / "frozen_choice.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
        for a in annotations:
            if a["split"] != split:
                continue
            try:
                obj = load_object(a, native[a["id"]], psf_match=not args.unmatched_psf)
                rows.extend(score_object(obj)); objects.append(obj)
                print(a["id"], split, obj["details"]["common_science_pixels"], "pixels", flush=True)
            except Exception as exc:
                failures.append({"id": a["id"], "split": split, "sample_group": a["sample_group"],
                                 "reason": f"{type(exc).__name__}: {exc}"})
    save_csv(OUT / "all_scores.csv", rows)
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (OUT / "coverage.json").write_text(json.dumps([o["details"] for o in objects], indent=2) + "\n")
    summary = summarize(rows, objects, frozen)
    summary["failures"] = failures
    summary["requested_strong_spirals"] = 24
    summary["requested_catalog_controls"] = 8
    summary["original_rendered_baseline"] = "Burned magenta telescope footprint removed by fixed color rule with filter guard; paired comparison only on identical surviving pixels. This baseline may fail coverage."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plots(objects, frozen, rows)
    print(json.dumps({"selected_pipeline": frozen["selected_pipeline"],
                      "test_AP": summary["test_macro_AP_by_pipeline"][frozen["selected_pipeline"]],
                      "selected_vs_raw_r": summary["selected_vs_raw_r"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
