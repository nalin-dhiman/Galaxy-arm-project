#!/usr/bin/env python3
"""Fixed first-development-object diagnostic of approximate PSF moment matching.

No image scores, heldout science images, or human arm masks are consulted.
Selection is the lowest frozen sample rank among development strong spirals.
This measures residual core/wing differences; it does not retune any kernel.

Run: MPLCONFIGDIR=/tmp/galaxy-mpl python experiments/psf_matching_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.filtering import ScienceImage, match_gaussian_psf

OUT = ROOT/"executed_filters/psf"
FWHM_PER_SIGMA = 2.354820045


def shape_measurements(data, pixel_scale):
    """Full positive stamp: centroid, trace moment, sorted-pixel cumulative flux."""
    a = np.asarray(data, float)
    if not np.isfinite(a).all():
        raise ValueError("PSF stamp must be finite")
    w = np.maximum(a, 0)
    if w.sum() <= 0:
        raise ValueError("PSF stamp has no positive light")
    w = w/w.sum()
    yy, xx = np.indices(w.shape)
    cx, cy = float((xx*w).sum()), float((yy*w).sum())
    dx, dy = xx-cx, yy-cy
    moment = float(((dx*dx+dy*dy)*w).sum()/2)
    radius = np.hypot(dx, dy)*pixel_scale
    order = np.argsort(radius.ravel(), kind="stable")
    sorted_radius = radius.ravel()[order]
    cumulative = np.cumsum(w.ravel()[order])
    cumulative /= cumulative[-1]
    radii = np.interp([.5, .8, .9], cumulative, sorted_radius)
    result = dict(centroid_x_px=cx, centroid_y_px=cy,
                  positive_sum=float(np.maximum(a, 0).sum()),
                  negative_absolute_fraction=float(np.maximum(-a, 0).sum()/np.maximum(abs(a).sum(), 1e-30)),
                  moment_variance_trace_over_two_px2=moment,
                  circular_moment_equivalent_fwhm_arcsec=float(FWHM_PER_SIGMA*np.sqrt(moment)*pixel_scale),
                  r50_arcsec=float(radii[0]), r80_arcsec=float(radii[1]), r90_arcsec=float(radii[2]))
    return result, (sorted_radius, cumulative), w


def main():
    manifest_path = ROOT/"executed_filters/native_images/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cohort = [r for r in manifest["images"]
              if r["split"] == "development" and r["sample_group"] == "strong_spiral"]
    first = min(cohort, key=lambda r: (int(r["rank"]), r["id"]))
    psf_path = ROOT/first["psf_path"]
    pixel_scale = float(first["psf"]["pixel_scale_arcsec"])
    stamps, measurements, curves = {}, {}, {}
    with fits.open(psf_path) as hdus:
        for band in "grz":
            hdu = int(first["psf"]["bands"][band]["hdu"])
            stamp = np.asarray(hdus[hdu].data, float)
            before, curve, positive = shape_measurements(stamp, pixel_scale)
            # Empty padding makes the kernel operate on a full zero background,
            # rather than reflecting a finite stamp's tiny edge wings inward.
            pad = max(stamp.shape)
            stamps[band] = np.pad(positive, pad, mode="constant")
            measurements[band] = dict(before=before)
            curves[band] = dict(before=curve)
    target = max(v["before"]["circular_moment_equivalent_fwhm_arcsec"] for v in measurements.values())
    for band in "grz":
        current = measurements[band]["before"]["circular_moment_equivalent_fwhm_arcsec"]
        image = ScienceImage(stamps[band], pixel_scale, psf_fwhm_arcsec=current,
                             unit="normalized PSF weight/pixel", pixel_value_kind="flux_per_pixel", band=band)
        matched = match_gaussian_psf(image, target)
        after, curve, positive = shape_measurements(matched.data, pixel_scale)
        if abs(after["positive_sum"]-1.) > 1e-8:
            raise RuntimeError("Padded PSF convolution unexpectedly lost flux")
        if abs(after["circular_moment_equivalent_fwhm_arcsec"]/target-1.) > 1e-4:
            raise RuntimeError("Gaussian quadrature broadening did not equalize second moments")
        measurements[band]["after"] = after
        measurements[band]["applied_gaussian_sigma_arcsec"] = float(np.sqrt(max(target**2-current**2, 0))/FWHM_PER_SIGMA)
        measurements[band]["moment_equivalent_fractional_target_difference"] = float(after["circular_moment_equivalent_fwhm_arcsec"]/target-1)
        curves[band]["after"] = curve
        stamps[band] = positive
    spreads = {}
    for key in ("circular_moment_equivalent_fwhm_arcsec", "r50_arcsec", "r80_arcsec", "r90_arcsec"):
        values = np.array([measurements[b]["after"][key] for b in "grz"])
        spreads[key] = dict(minimum=float(values.min()), maximum=float(values.max()),
                            max_minus_min_over_mean=float(np.ptp(values)/values.mean()))
    summary = dict(id=first["id"], frozen_sample_rank=int(first["rank"]),
                   split="development", sample_group="strong_spiral",
                   selection="Lowest frozen rank among development strong spirals; no image-score selection",
                   psf_source_url=first["psf_source_url"], psf_path=str(psf_path.relative_to(ROOT)),
                   psf_sha256=hashlib.sha256(psf_path.read_bytes()).hexdigest(),
                   native_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   pixel_scale_arcsec=pixel_scale, target_moment_equivalent_fwhm_arcsec=target,
                   measurement="Full positive finite PSF stamp, normalized; positive weights only; sorted pixel-center cumulative radii",
                   kernel="Same circular Gaussian quadrature broadening as primary benchmark; no parameter adjustment",
                   bands=measurements, after_matching_spreads=spreads,
                   interpretation="Near equality of circular second moments does not equalize non-Gaussian PSF cores or wings.",
                   limitations=["One deterministic development example, not a population PSF study.",
                                "The finite local PSF stamp is an instrumental model at the target center.",
                                "Encircled-light radii use discrete pixel centers, so subpixel precision is not claimed.",
                                "Negative PSF sidelobes are excluded from shape measurements and recorded explicitly.",
                                "No heldout filter scores, arm labels, or science-image brightness measurements enter this diagnostic."])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"quantitative.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    colors = dict(g="tab:blue", r="tab:orange", z="tab:green")
    for j, band in enumerate("grz"):
        w = stamps[band]
        cy, cx = np.array(w.shape)//2
        half = int(np.ceil(3.2/pixel_scale))
        cut = w[cy-half:cy+half+1, cx-half:cx+half+1]
        ax = axes[0, j]
        ax.imshow(np.log10(np.maximum(cut/w.max(), 1e-5)), origin="lower", cmap="magma",
                  vmin=-4, vmax=0, extent=[-3.2, 3.2, -3.2, 3.2])
        ax.set(title=f"{band}: broadened PSF / its peak (log10)", xlabel="Offset [arcsec]", ylabel="Offset [arcsec]")
    for band in "grz":
        for state, style in (("before", "--"), ("after", "-")):
            rr, ff = curves[band][state]
            axes[1, 0].plot(rr, ff, style, color=colors[band], label=f"{band} {state}")
    axes[1, 0].set(xlim=(0, 3), ylim=(0, 1), xlabel="Radius [arcsec]", ylabel="Positive encircled PSF fraction",
                   title="Equal moments leave unequal core / wing shapes")
    axes[1, 0].legend(fontsize=7)
    keys = ("r50_arcsec", "r80_arcsec", "r90_arcsec")
    for j, band in enumerate("grz"):
        axes[1, 1].bar(np.arange(3)+(j-1)*.24,
                        [measurements[band]["after"][k] for k in keys], width=.24, label=band, color=colors[band])
    axes[1, 1].set(xticks=np.arange(3), xticklabels=["R50", "R80", "R90"],
                   ylabel="Radius [arcsec]", title="Radii after fixed moment broadening")
    axes[1, 1].legend()
    axes[1, 2].axis("off")
    lines = [f"Object: {first['id']} (first development)",
             f"Target equivalent scale: {target:.3f} arcsec", "", "Matched equivalent FWHM / R50 / R80:"]
    for band in "grz":
        a = measurements[band]["after"]
        lines.append(f"{band}: {a['circular_moment_equivalent_fwhm_arcsec']:.3f} / {a['r50_arcsec']:.3f} / {a['r80_arcsec']:.3f}")
    lines += ["", "Full PSF equality is not established.", "Do not assign all band differences to color.", "No filter scores or arm masks were inspected."]
    axes[1, 2].text(0, .95, "\n".join(lines), va="top", fontsize=10)
    fig.suptitle("Local instrumental PSF diagnostic: Gaussian moment matching is an approximation", fontsize=14)
    fig.savefig(OUT/"diagnostic.png", dpi=165)
    plt.close(fig)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
